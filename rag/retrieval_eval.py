import argparse
import importlib.util
import json
from pathlib import Path

_BOOTSTRAP_PATH = Path(__file__).resolve().parents[1] / "bootstrap.py"
_BOOTSTRAP_SPEC = importlib.util.spec_from_file_location("aimanga_bootstrap", _BOOTSTRAP_PATH)
if _BOOTSTRAP_SPEC is None or _BOOTSTRAP_SPEC.loader is None:
    raise ImportError(f"无法加载 bootstrap: {_BOOTSTRAP_PATH}")
_bootstrap = importlib.util.module_from_spec(_BOOTSTRAP_SPEC)
_BOOTSTRAP_SPEC.loader.exec_module(_bootstrap)
_bootstrap.ensure_project_root()
from rag.rag_service import RagSummarizeService


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="运行本地 RAG 管道的检索评估")
    parser.add_argument(
        "--dataset",
        default="eval/retrieval_eval_samples.yml",
        help="Evaluation dataset path relative to the project root.",
    )
    parser.add_argument(
        "--mode",
        default="all",
        choices=["baseline", "enhanced", "enhanced_retry", "all"],
        help="Evaluation mode.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    service = RagSummarizeService()
    modes = ["baseline", "enhanced", "enhanced_retry"] if args.mode == "all" else [args.mode]

    for mode in modes:
        report = service.evaluate_retrieval(args.dataset, mode=mode)
        print(f"=== {mode} ===")
        print(json.dumps(report.metrics, ensure_ascii=False, indent=2))
        print()


if __name__ == "__main__":
    main()
