#!/usr/bin/env python3
import argparse
import subprocess
import sys
import os
import glob
import json
from datetime import datetime

def _latest_summary(dir_):
    files = glob.glob(os.path.join(dir_, "*_summary.json"))
    if not files:
        raise FileNotFoundError(f"No *_summary.json found in {dir_}.")
    files.sort(key=os.path.getmtime, reverse=True)
    with open(files[0], "r") as f:
        return json.load(f), files[0]

def main():
    p = argparse.ArgumentParser(description="Run AIME25 in standard + branching and combine results.")
    p.add_argument("--runfile", default="run_aime25_full.py", help="Path to the single-run script.")
    p.add_argument("--model", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B")
    p.add_argument("--budget", type=int, default=16, help="Standard mode traces.")
    p.add_argument("--initial_branches", type=int, default=4)
    p.add_argument("--max_total_branches", type=int, default=16)
    p.add_argument("--tensor_parallel_size", type=int, default=4)
    p.add_argument("--subset", choices=["AIME2025-I", "AIME2025-II"], default=None)
    p.add_argument("--output_dir", default="results/aime25_full", help="Parent output directory.")
    p.add_argument("--resume", default=None)
    p.add_argument("--dry_run", action="store_true")
    args = p.parse_args()

    base_out = args.output_dir
    std_out = os.path.join(base_out, "standard")
    br_out  = os.path.join(base_out, "branching")
    os.makedirs(std_out, exist_ok=True)
    os.makedirs(br_out,  exist_ok=True)

    def common(run_out):
        cmd = [
            sys.executable, args.runfile,
            "--model", args.model,
            "--tensor_parallel_size", str(args.tensor_parallel_size),
            "--output_dir", run_out
        ]
        if args.subset:
            cmd += ["--subset", args.subset]
        if args.resume:
            cmd += ["--resume", args.resume]
        if args.dry_run:
            cmd += ["--dry_run"]
        return cmd

    # 1) Standard
    std_cmd = common(std_out) + ["--mode", "standard", "--budget", str(args.budget)]
    subprocess.run(std_cmd, check=True)

    # 2) Branching
    br_cmd = common(br_out) + [
        "--mode", "branching",
        "--initial_branches", str(args.initial_branches),
        "--max_total_branches", str(args.max_total_branches),
    ]
    subprocess.run(br_cmd, check=True)

    std_summary, std_path = _latest_summary(std_out)
    br_summary,  br_path  = _latest_summary(br_out)

    combined = {
        "created_at": datetime.now().isoformat(),
        "runs": {
            "standard": {
                "summary": std_summary,
                "summary_path": os.path.relpath(std_path, base_out),
                "output_dir": os.path.relpath(std_out,  base_out)
            },
            "branching": {
                "summary": br_summary,
                "summary_path": os.path.relpath(br_path, base_out),
                "output_dir": os.path.relpath(br_out, base_out)
            }
        }
    }

    combined_path = os.path.join(base_out, "combined_summary.json")
    with open(combined_path, "w") as f:
        json.dump(combined, f, indent=2)
    print(f"\nCombined summary written to: {combined_path}")

if __name__ == "__main__":
    main()
