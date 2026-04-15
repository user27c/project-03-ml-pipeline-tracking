#!/usr/bin/env python3
"""
FR-3 回滚工具：将指定模型版本切换到 Production（并归档当前 Production）。

用法：
  python scripts/rollback_model.py --version 3

可选：
  MLFLOW_TRACKING_URI=http://localhost:5000
"""

from __future__ import annotations

import argparse
import os

import mlflow


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--name", default="image_classifier")
    p.add_argument("--version", required=True)
    p.add_argument("--stage", default="Production")
    p.add_argument("--archive-existing", action="store_true", default=True)
    args = p.parse_args()

    uri = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    mlflow.set_tracking_uri(uri)
    client = mlflow.tracking.MlflowClient()
    client.transition_model_version_stage(
        name=args.name,
        version=str(args.version),
        stage=args.stage,
        archive_existing_versions=bool(args.archive_existing),
    )
    print(f"OK: {args.name} v{args.version} -> {args.stage} (archive_existing={args.archive_existing})")


if __name__ == "__main__":
    main()

