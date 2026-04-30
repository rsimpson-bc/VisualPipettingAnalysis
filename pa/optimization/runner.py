"""
Optimization harness — runs the PA pipeline against stored run data.

This module is completely standalone: no HTTP server, no Piomek2, no live
instrument. It reads stored images + expected results and produces scored
outputs, enabling automated parameter search.

Usage:
    python -m pa.optimization.runner \\
        --run-folder  C:/runs/run_20260427_143512 \\
        --instrument-config C:/configs/instrument_config.json \\
        --analysis-config  C:/configs/analysis_config.json \\
        --pipeline full_liquid \\
        --step-ids step_0001 step_0002

The runner:
  1. Loads all stored images and expected values from the run folder.
  2. Runs the requested pipeline on each step.
  3. Writes raw results to stdout / a CSV.
  4. Calls scorer.py to compare against expected values.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List, Optional

from pa.state import loaders
from pa.optimization.step_runner import run_step
from pa.optimization.scorer import score_results


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="PA parameter optimization runner")
    parser.add_argument("--run-folder", required=True)
    parser.add_argument("--instrument-config", required=True)
    parser.add_argument("--analysis-config", required=True)
    parser.add_argument("--pipeline", required=True,
                        help="Named pipeline from analysis_config.json")
    parser.add_argument("--step-ids", nargs="+", default=None,
                        help="Steps to evaluate (default: all in run folder)")
    parser.add_argument("--output", default=None,
                        help="Path for CSV score output (default: stdout)")
    args = parser.parse_args(argv)

    instrument_config = loaders.load_instrument_config(args.instrument_config)
    with open(args.analysis_config, encoding="utf-8") as f:
        analysis_config = json.load(f)

    pipeline_config = analysis_config["pipelines"].get(args.pipeline)
    if pipeline_config is None:
        print(f"ERROR: pipeline '{args.pipeline}' not found in analysis_config.", file=sys.stderr)
        sys.exit(1)

    # Discover steps
    step_ids = args.step_ids or _discover_steps(args.run_folder)
    if not step_ids:
        print("No steps found.", file=sys.stderr)
        sys.exit(1)

    all_scores = []
    for step_id in step_ids:
        print(f"Running {step_id}...")
        result, expected = run_step(
            step_id=step_id,
            run_folder=args.run_folder,
            instrument_config=instrument_config,
            pipeline_config=pipeline_config,
        )
        score = score_results(result, expected)
        all_scores.append({"step_id": step_id, **score})
        print(f"  {step_id}: {score}")

    # Output
    if args.output:
        import csv
        with open(args.output, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=all_scores[0].keys())
            writer.writeheader()
            writer.writerows(all_scores)
        print(f"Scores written to {args.output}")
    else:
        for row in all_scores:
            print(row)


def _discover_steps(run_folder: str) -> List[str]:
    """Find step IDs from existing step_meta.json files under the run folder."""
    step_ids = []
    for root, _dirs, files in os.walk(run_folder):
        for fname in files:
            if fname == "step_meta.json":
                with open(os.path.join(root, fname), encoding="utf-8") as f:
                    meta = json.load(f)
                if "step_id" in meta:
                    step_ids.append(meta["step_id"])
    return sorted(set(step_ids))


if __name__ == "__main__":
    main()
