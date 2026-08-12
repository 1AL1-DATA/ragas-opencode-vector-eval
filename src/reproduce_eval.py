"""
End-to-end reproduction driver for the evaluation pipeline.

Stages (each is a thin wrapper around a module in this package):
  1. dataset        - build the sampled dataset from the vector store
                      (needs the private opencode session corpus; see data/README.md)
  2. ragas          - run the 4 RAGAS metrics with a local LLM judge (hours)
  3. shortcuts      - run the local shortcut suite (minutes)
  4. compare        - correlate RAGAS vs shortcuts, write comparison artifacts
  5. aggregates     - write this repo's public results/ and figures/

Usage:
  python -m src.reproduce_eval --stage shortcuts
  python -m src.reproduce_eval          # full pipeline
"""
import argparse
import subprocess
import sys

STAGES = ["dataset", "ragas", "shortcuts", "compare", "aggregates"]


def run(mod, *args):
    print(f"\n=== stage: {mod} ===\n")
    subprocess.run([sys.executable, "-m", f"src.{mod}", *args], check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=STAGES, help="run a single stage")
    args = ap.parse_args()

    if args.stage == "dataset":
        run("build_eval_dataset")
    elif args.stage == "ragas":
        run("run_ragas")
    elif args.stage == "shortcuts":
        run("shortcut_metrics")
    elif args.stage == "compare":
        run("compare_eval")
    elif args.stage == "aggregates":
        run("build_aggregates")
    else:
        for s in STAGES:
            run(s)
    print("\ndone")


if __name__ == "__main__":
    main()
