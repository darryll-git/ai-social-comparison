"""
calibrate.py - Automated baseline calibration CLI for AI Social Comparison experiment.

Executes baseline trials across 3 prompt trade-off intensities (low=5%, medium=15%, high=25%
quality penalty) and automatically selects the parameter setting that lands baseline acceptance
rate closest to the 50% target equilibrium (within the 40%-60% dilemmic window).

Usage:
  python calibrate.py --dry-run
  python calibrate.py --models "openai/gpt-4o-mini"
  python calibrate.py --tier ultra-low
"""

import sys
import experiment

if __name__ == "__main__":
    # Ensure --auto-calibrate flag is present in arguments if not explicitly supplied
    if "--auto-calibrate" not in sys.argv and "--mode" not in sys.argv:
        sys.argv.insert(1, "--auto-calibrate")
    experiment.main()
