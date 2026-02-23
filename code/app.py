import argparse
import subprocess
import sys
import time

STEPS = {
    "eda":           "code/data_exploration.py",
    "preprocessing": "code/preprocessing.py",
    "modeling":      "code/modeling.py",
}

def run_step(name: str, script: str):
    print(f"\n{'='*60}")
    print(f"  RUNNING STEP: {name.upper()}")
    print(f"  Script      : {script}")
    print(f"{'='*60}\n")
    start = time.time()
    result = subprocess.run([sys.executable, script], check=False)
    elapsed = time.time() - start
    if result.returncode == 0:
        print(f"\n✅  {name} completed in {elapsed:.1f}s")
    else:
        print(f"\n❌  {name} FAILED (exit code {result.returncode})")
        sys.exit(result.returncode)

def main():
    parser = argparse.ArgumentParser(description="Diabetes ML Pipeline")
    parser.add_argument(
        "--step", choices=list(STEPS.keys()), default=None,
        help="Run a single step (default: run all steps)"
    )
    args = parser.parse_args()

    print("\n" + "█" * 60)
    print("  DIABETES ML PROJECT — FULL PIPELINE")
    print("█" * 60)

    if args.step:
        run_step(args.step, STEPS[args.step])
    else:
        for name, script in STEPS.items():
            run_step(name, script)
        print("\n" + "█" * 60)
        print("  🎉  ALL STEPS COMPLETE!")
        print("  📊  MLflow: mlflow ui  →  http://127.0.0.1:5000")
        print("  📁  Reports: reports/figures/")
        print("█" * 60 + "\n")

if __name__ == "__main__":
    main()