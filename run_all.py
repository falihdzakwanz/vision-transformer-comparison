"""
Run All Pipeline
================
Script untuk menjalankan seluruh pipeline dari data preparation hingga visualization.

Usage:
    python run_all.py [--skip-training]

Options:
    --skip-training    Skip training phase (useful if models already trained)

Author: 122140132 - Falih Dzakwan Zuhdi
Course: Deep Learning - Semester Ganjil 2025/2026
"""

import sys
import time
import subprocess
from pathlib import Path
import argparse


def print_header(text):
    """Print formatted header."""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70 + "\n")


def run_script(script_path, description):
    """
    Run a Python script and handle errors.

    Args:
        script_path (Path): Path to script
        description (str): Description of what the script does

    Returns:
        bool: True if successful, False otherwise
    """
    print_header(f"RUNNING: {description}")
    print(f"Script: {script_path}\n")

    start_time = time.time()

    try:
        result = subprocess.run(
            [sys.executable, str(script_path)], check=True, capture_output=False
        )

        elapsed = time.time() - start_time
        print(f"\n✓ Completed in {elapsed:.1f} seconds")
        return True

    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        print(f"\n✗ Failed after {elapsed:.1f} seconds")
        print(f"Error: {e}")
        return False
    except KeyboardInterrupt:
        print("\n\n⚠ Interrupted by user")
        return False


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Run complete Vision Transformer comparison pipeline"
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip training phase (use if models already trained)",
    )

    args = parser.parse_args()

    # Define base directory
    base_dir = Path(__file__).parent
    src_dir = base_dir / "src"

    # Pipeline steps
    steps = [
        {
            "script": src_dir / "data_preparation.py",
            "description": "Data Preparation and Exploration",
            "skippable": False,
        },
        {
            "script": src_dir / "train_swin.py",
            "description": "Training Swin Transformer",
            "skippable": True,
        },
        {
            "script": src_dir / "train_deit.py",
            "description": "Training DeiT",
            "skippable": True,
        },
        {
            "script": src_dir / "evaluate.py",
            "description": "Model Evaluation",
            "skippable": False,
        },
        {
            "script": src_dir / "visualize.py",
            "description": "Results Visualization",
            "skippable": False,
        },
    ]

    # Print welcome message
    print("\n" + "=" * 70)
    print(" " * 15 + "VISION TRANSFORMER COMPARISON PIPELINE")
    print(" " * 20 + "Swin Transformer vs DeiT")
    print("=" * 70)

    print("\nThis will run the complete pipeline:")
    print("  1. Data Preparation")
    if not args.skip_training:
        print("  2. Train Swin Transformer")
        print("  3. Train DeiT")
    else:
        print("  2. [SKIPPED] Train Swin Transformer")
        print("  3. [SKIPPED] Train DeiT")
    print("  4. Evaluate Models")
    print("  5. Visualize Results")

    if not args.skip_training:
        print("\n⚠ Warning: Training will take 1-2 hours (with GPU)")
        response = input("\nContinue? (y/n): ")
        if response.lower() != "y":
            print("Cancelled by user")
            return

    # Start pipeline
    total_start = time.time()
    failed_steps = []

    for i, step in enumerate(steps, 1):
        # Skip training steps if requested
        if args.skip_training and step["skippable"]:
            print_header(f"STEP {i}/{len(steps)}: {step['description']} [SKIPPED]")
            continue

        print_header(f"STEP {i}/{len(steps)}: {step['description']}")

        success = run_script(step["script"], step["description"])

        if not success:
            failed_steps.append(step["description"])
            print("\n⚠ Would you like to continue with remaining steps?")
            response = input("Continue? (y/n): ")
            if response.lower() != "y":
                break

    # Final summary
    total_elapsed = time.time() - total_start

    print("\n" + "=" * 70)
    print("  PIPELINE EXECUTION SUMMARY")
    print("=" * 70)

    print(f"\nTotal execution time: {total_elapsed/60:.1f} minutes")

    if not failed_steps:
        print("\n✓ All steps completed successfully!")
        print("\nGenerated outputs:")
        print("  📊 Results: outputs/results/")
        print("  📈 Figures: outputs/figures/")
        print("  🤖 Models: models/")

        print("\nNext steps:")
        print("  1. Review figures in outputs/figures/")
        print("  2. Check comparison table in outputs/results/model_comparison.csv")
        print("  3. Fill LaTeX report with results")
        print("  4. Upload code to GitHub")
        print("  5. Submit PDF to Moodle")

    else:
        print("\n⚠ Some steps failed:")
        for step in failed_steps:
            print(f"  ✗ {step}")

        print("\nPlease check the errors and try running individual scripts:")
        print("  python src/data_preparation.py")
        print("  python src/train_swin.py")
        print("  python src/train_deit.py")
        print("  python src/evaluate.py")
        print("  python src/visualize.py")

    print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠ Pipeline interrupted by user")
        print("You can resume by running individual scripts in src/")
        sys.exit(1)
