"""
CLI script for automated downloading, extraction, and verification of BEIR benchmark datasets.
"""
import argparse
from pathlib import Path
import sys

# Ensure root workspace is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.loader import BEIR_URLS, BEIRDatasetLoader
from src.utils.logger import setup_logger

logger = setup_logger("AdaptiveRetriever.DownloadData")


def parse_args():
    parser = argparse.ArgumentParser(description="Download BEIR datasets for AdaptiveRetriever")
    parser.add_argument(
        "--datasets", "--dataset",
        nargs="+",
        default=["scifact"],
        help=f"Dataset name(s) to download. Choices: {list(BEIR_URLS.keys())}",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data",
        help="Local target directory for dataset storage",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if dataset directory exists",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    loader = BEIRDatasetLoader(data_dir=args.data_dir)

    logger.info(f"Target data directory: {Path(args.data_dir).resolve()}")
    for ds in args.datasets:
        ds_name = ds.lower().strip()
        logger.info(f"Processing dataset: '{ds_name}'...")
        try:
            split_data = loader.load_split(ds_name, split="test", force_download=args.force)
            logger.info(
                f"Successfully loaded and verified '{ds_name}': "
                f"{len(split_data.corpus):,} documents, "
                f"{len(split_data.queries):,} queries, "
                f"{sum(len(v) for v in split_data.qrels.values()):,} qrel judgments."
            )
        except Exception as e:
            logger.error(f"Failed to download/verify dataset '{ds_name}': {e}")
            sys.exit(1)

    logger.info("All requested datasets downloaded and verified successfully.")


if __name__ == "__main__":
    main()
