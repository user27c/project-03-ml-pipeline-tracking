"""
FR-2.2 超参数调优（网格搜索版）

目标：
- 运行 >=10 组配置
- 每组都在 MLflow 中作为一个 Run（nested）记录参数/指标/模型/工件
- 输出 best config（按 best_val_accuracy 最高）

注意：
- 为了复用现有训练实现，本模块会创建 nested runs，并调用 ModelTrainer.train(manage_mlflow_run=False)
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import mlflow

from src.dataset import create_data_loaders
from src.training import MLflowTracker, ModelTrainer


def _grid_space() -> Dict[str, List[Any]]:
    # 与 requirements.md 示例空间对齐（最少 10 组可从 product 截取）
    return {
        "learning_rate": [0.0001, 0.001, 0.01],
        "batch_size": [16, 32, 64],
        "model_name": ["resnet18", "mobilenet_v2"],
        "optimizer": ["adam", "sgd"],
    }


def _iter_configs(max_trials: int) -> List[Dict[str, Any]]:
    space = _grid_space()
    keys = list(space.keys())
    combos = list(itertools.product(*(space[k] for k in keys)))
    out: List[Dict[str, Any]] = []
    for values in combos[: max_trials]:
        cfg = dict(zip(keys, values))
        out.append(cfg)
    return out


def _default_tracking_uri() -> str:
    return os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")


def run_tuning(
    csv_path: str,
    root_dir: str,
    experiment_name: str,
    n_trials: int,
    num_epochs: int,
    lr_step_size: int,
    lr_gamma: float,
    early_stopping_patience: int,
    seed: int,
    deterministic: bool,
) -> Tuple[Dict[str, Any], float]:
    tracking_uri = _default_tracking_uri()
    tracker = MLflowTracker(tracking_uri=tracking_uri, experiment_name=experiment_name)

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    best_cfg: Dict[str, Any] = {}
    best_val = -1.0

    parent_name = f"tune_grid_{Path(csv_path).name}"
    with mlflow.start_run(run_name=parent_name, tags={"task": "hparam_tuning", "method": "grid"}):
        configs = _iter_configs(n_trials)
        mlflow.log_param("tuning_n_trials", len(configs))
        mlflow.log_param("tuning_method", "grid")
        mlflow.log_param("tuning_space", json.dumps(_grid_space(), ensure_ascii=False))

        for i, cfg in enumerate(configs):
            run_name = f"trial_{i+1:02d}_{cfg['model_name']}_bs{cfg['batch_size']}_lr{cfg['learning_rate']}_{cfg['optimizer']}"
            with mlflow.start_run(run_name=run_name, nested=True):
                params = {
                    **cfg,
                    "num_epochs": int(num_epochs),
                    "lr_step_size": int(lr_step_size),
                    "lr_gamma": float(lr_gamma),
                    "early_stopping_patience": int(early_stopping_patience),
                    "seed": int(seed),
                    "deterministic": bool(deterministic),
                }

                train_loader, val_loader, _ = create_data_loaders(
                    csv_path=csv_path,
                    root_dir=root_dir,
                    batch_size=int(params["batch_size"]),
                    num_workers=0,
                    pin_memory=False,
                )
                num_classes = train_loader.dataset.num_classes

                trainer = ModelTrainer(
                    config={
                        # 复用 dags PIPELINE_CONFIG 同名 key
                        "model_save_path": str(Path("models")),
                        "artifacts_path": str(Path("artifacts")),
                    },
                    mlflow_tracker=tracker,
                )

                _, best_val_acc = trainer.train(
                    train_loader=train_loader,
                    val_loader=val_loader,
                    num_classes=num_classes,
                    params=params,
                    manage_mlflow_run=False,
                )

                # trial-level summary
                mlflow.log_metric("trial_best_val_accuracy", float(best_val_acc))

                if float(best_val_acc) > best_val:
                    best_val = float(best_val_acc)
                    best_cfg = params

        # parent summary artifact
        out_dir = Path("artifacts") / "tuning"
        out_dir.mkdir(parents=True, exist_ok=True)
        best_path = out_dir / "best_config.json"
        best_path.write_text(json.dumps({"best_val_accuracy": best_val, "params": best_cfg}, indent=2, ensure_ascii=False), encoding="utf-8")
        mlflow.log_artifact(str(best_path))
        mlflow.log_metric("best_val_accuracy", float(best_val))

    return best_cfg, best_val


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-path", default="data/processed", help="兼容 MLproject；实际需传 all_splits.csv 路径")
    p.add_argument("--csv-path", default=None, help="推荐：直接传 all_splits.csv / all_splits.fast.csv")
    p.add_argument("--root-dir", default=".", help="图像路径相对根目录（与 Airflow PROJECT_ROOT 一致）")
    p.add_argument("--experiment-name", default=os.environ.get("MLFLOW_EXPERIMENT_NAME", "image_classification_pipeline"))
    p.add_argument("--n-trials", type=int, default=10)
    p.add_argument("--num-epochs", type=int, default=2)
    p.add_argument("--lr-step-size", type=int, default=5)
    p.add_argument("--lr-gamma", type=float, default=0.1)
    p.add_argument("--early-stopping-patience", type=int, default=1)
    p.add_argument("--seed", type=int, default=int(os.environ.get("ML_PIPELINE_SEED", "42")))
    p.add_argument("--deterministic", type=str, default=os.environ.get("ML_PIPELINE_DETERMINISTIC", "true"))
    args = p.parse_args()

    csv_path = args.csv_path
    if not csv_path:
        # MLproject 传的是 data_path，这里给一个宽松兜底：假设目录下有 all_splits.csv
        csv_path = str(Path(args.data_path) / "all_splits.csv")

    det = str(args.deterministic).lower() in ("1", "true", "yes")
    best_cfg, best_val = run_tuning(
        csv_path=csv_path,
        root_dir=args.root_dir,
        experiment_name=args.experiment_name,
        n_trials=args.n_trials,
        num_epochs=args.num_epochs,
        lr_step_size=args.lr_step_size,
        lr_gamma=args.lr_gamma,
        early_stopping_patience=args.early_stopping_patience,
        seed=args.seed,
        deterministic=det,
    )
    print("Best val accuracy:", best_val)
    print("Best params:", json.dumps(best_cfg, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

