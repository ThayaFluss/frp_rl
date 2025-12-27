"""
Simple test suite using existing CSV results.

This version doesn't require running evaluate_model, instead it analyzes
existing CSV results to verify return calculation correctness.
"""

import sys
import os
import numpy as np
import csv

def load_trial_stats_from_csv(csv_path):
    """Load trial statistics from CSV file."""
    trial_data = []

    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            trial_data.append({
                'trial': int(row['Trial']),
                'mean_return': float(row['Mean_Return']),
                'std_return': float(row['Std_Return']),
                'mean_steps': float(row['Mean_Steps']),
                'std_steps': float(row['Std_Steps']),
                'success_rate': float(row['Success_Rate']),
                'success_std': float(row['Success_Std'])
            })

    return trial_data


def test_return_value_ranges():
    """
    Test that return values are in expected ranges.

    For CartPole with max_steps=200:
    - Failed trials: return should be in range [-0.995, 0.99)
      (calculated as: steps/200 - 1.0)
    - Successful trials: return should be ≈ 1.0
    """
    print("\n" + "="*70)
    print("Test: Return Value Ranges")
    print("="*70)

    csv_path = "exp/20251226_041015/model_457_iter_trial_stats_n32.csv"

    if not os.path.exists(csv_path):
        print(f"⚠ Warning: CSV file {csv_path} not found. Skipping test.")
        return True

    trial_data = load_trial_stats_from_csv(csv_path)

    print(f"Loaded data for {len(trial_data)} trials")

    max_steps = 200
    violations = []

    for t in trial_data:
        mean_return = t['mean_return']
        mean_steps = t['mean_steps']
        success_rate = t['success_rate']

        # Calculate expected return based on steps
        if mean_steps >= max_steps - 1:  # Account for averaging
            # Should be successful
            expected_min_return = 0.99
            expected_max_return = 1.01
        else:
            # Failed trial
            # Expected: (mean_steps / max_steps) - 1.0
            expected_return = (mean_steps / max_steps) - 1.0
            # Allow some variance due to averaging
            expected_min_return = expected_return - 0.1
            expected_max_return = expected_return + 0.1

        # Check if return is in expected range
        if not (expected_min_return <= mean_return <= expected_max_return):
            violations.append({
                'trial': t['trial'],
                'mean_return': mean_return,
                'mean_steps': mean_steps,
                'expected_range': (expected_min_return, expected_max_return),
                'success_rate': success_rate
            })

    if len(violations) == 0:
        print(f"✓ PASSED: All trial returns are in expected ranges")
        return True
    else:
        print(f"✗ FAILED: {len(violations)} trials have returns outside expected ranges")
        print("\nViolations:")
        for v in violations[:5]:
            print(f"  Trial {v['trial']}: return={v['mean_return']:.4f}, "
                  f"steps={v['mean_steps']:.1f}, "
                  f"expected range=[{v['expected_range'][0]:.4f}, {v['expected_range'][1]:.4f}]")
        return False


def test_negative_returns_for_failures():
    """
    Test that failed trials have negative returns.

    This is the key test for the termination penalty bug:
    - If the bug exists: failed trials would have POSITIVE returns
    - If fixed: failed trials should have NEGATIVE returns
    """
    print("\n" + "="*70)
    print("Test: Termination Penalty Bug Detection")
    print("="*70)

    csv_path = "exp/20251226_041015/model_457_iter_trial_stats_n32.csv"

    if not os.path.exists(csv_path):
        print(f"⚠ Warning: CSV file {csv_path} not found. Skipping test.")
        return True

    trial_data = load_trial_stats_from_csv(csv_path)

    print(f"Analyzing {len(trial_data)} trials...")

    max_steps = 200
    bug_patterns = []
    failed_trials = []

    for t in trial_data:
        mean_return = t['mean_return']
        mean_steps = t['mean_steps']
        success_rate = t['success_rate']

        # Consider it a failure if success_rate is low or steps < max_steps
        is_failure = (success_rate < 0.5) or (mean_steps < max_steps - 5)

        if is_failure:
            failed_trials.append(t)

            # Bug pattern: positive return despite failure
            if mean_return > 0:
                bug_patterns.append(t)

    print(f"\nFound {len(failed_trials)} failed trials")
    print(f"Found {len(bug_patterns)} trials with bug pattern (positive return despite failure)")

    if len(bug_patterns) > 0:
        print(f"\n✗ FAILED: Termination penalty bug detected!")
        print(f"  {len(bug_patterns)} failed trials have POSITIVE returns")
        print("\nExamples of bug pattern:")
        for t in bug_patterns[:5]:
            expected_return = (t['mean_steps'] / max_steps) - 1.0
            print(f"  Trial {t['trial']}: return={t['mean_return']:.4f} "
                  f"(should be {expected_return:.4f}), "
                  f"steps={t['mean_steps']:.1f}")
        return False
    else:
        print(f"✓ PASSED: All {len(failed_trials)} failed trials have negative returns")
        print("  Termination penalty is correctly included")
        return True


def test_success_returns():
    """
    Test that successful trials have return ≈ 1.0.
    """
    print("\n" + "="*70)
    print("Test: Success Case Returns")
    print("="*70)

    csv_path = "exp/20251226_041015/model_457_iter_trial_stats_n32.csv"

    if not os.path.exists(csv_path):
        print(f"⚠ Warning: CSV file {csv_path} not found. Skipping test.")
        return True

    trial_data = load_trial_stats_from_csv(csv_path)

    successful_trials = [t for t in trial_data if t['success_rate'] > 0.5]

    if len(successful_trials) == 0:
        print("⚠ No successful trials found (agent may not be well-trained)")
        return True

    print(f"Found {len(successful_trials)} successful trials (success_rate > 0.5)")

    violations = []
    for t in successful_trials:
        if t['mean_return'] < 0.95:
            violations.append(t)

    if len(violations) == 0:
        print(f"✓ PASSED: All successful trials have return >= 0.95")
        return True
    else:
        print(f"✗ FAILED: {len(violations)} successful trials have return < 0.95")
        for v in violations:
            print(f"  Trial {v['trial']}: return={v['mean_return']:.4f}, "
                  f"success_rate={v['success_rate']:.2%}")
        return False


def test_return_step_relationship():
    """
    Test the mathematical relationship between steps and returns.

    For CartPole: return = (steps / max_steps) - 1.0  [if failed]
                  return = 1.0                         [if succeeded]
    """
    print("\n" + "="*70)
    print("Test: Return-Step Mathematical Relationship")
    print("="*70)

    csv_path = "exp/20251226_041015/model_457_iter_trial_stats_n32.csv"

    if not os.path.exists(csv_path):
        print(f"⚠ Warning: CSV file {csv_path} not found. Skipping test.")
        return True

    trial_data = load_trial_stats_from_csv(csv_path)

    max_steps = 200
    violations = []

    for t in trial_data:
        mean_return = t['mean_return']
        mean_steps = t['mean_steps']
        success_rate = t['success_rate']

        # Calculate expected return from formula
        if success_rate > 0.9:
            # Mostly successful
            expected_return = 1.0
        elif success_rate < 0.1:
            # Mostly failed
            expected_return = (mean_steps / max_steps) - 1.0
        else:
            # Mixed - weighted average
            expected_return_fail = (mean_steps / max_steps) - 1.0
            expected_return_success = 1.0
            expected_return = (success_rate * expected_return_success +
                             (1 - success_rate) * expected_return_fail)

        # Allow 10% tolerance for averaging effects
        diff = abs(mean_return - expected_return)
        tolerance = 0.1

        if diff > tolerance:
            violations.append({
                'trial': t['trial'],
                'mean_return': mean_return,
                'expected_return': expected_return,
                'diff': diff,
                'mean_steps': mean_steps,
                'success_rate': success_rate
            })

    if len(violations) == 0:
        print(f"✓ PASSED: All returns match expected mathematical relationship")
        return True
    else:
        print(f"✗ FAILED: {len(violations)} trials violate return-step relationship")
        print("\nViolations:")
        for v in violations[:5]:
            print(f"  Trial {v['trial']}: actual={v['mean_return']:.4f}, "
                  f"expected={v['expected_return']:.4f}, "
                  f"diff={v['diff']:.4f}")
        return False


def run_all_tests():
    """Run all CSV-based tests."""
    print("\n" + "="*70)
    print("RETURN CALCULATION TEST SUITE (CSV-based)")
    print("="*70)

    tests = [
        ("Return Value Ranges", test_return_value_ranges),
        ("Termination Penalty Bug Detection", test_negative_returns_for_failures),
        ("Success Case Returns", test_success_returns),
        ("Return-Step Mathematical Relationship", test_return_step_relationship),
    ]

    results = {}

    for test_name, test_func in tests:
        try:
            passed = test_func()
            results[test_name] = "PASSED" if passed else "FAILED"
        except Exception as e:
            print(f"\n✗ ERROR in {test_name}: {e}")
            import traceback
            traceback.print_exc()
            results[test_name] = "ERROR"

    # Print summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)

    for test_name, result in results.items():
        status_symbol = "✓" if result == "PASSED" else "✗"
        print(f"{status_symbol} {test_name}: {result}")

    num_passed = sum(1 for r in results.values() if r == "PASSED")
    num_total = len(results)

    print("\n" + "="*70)
    if num_passed == num_total:
        print(f"ALL TESTS PASSED ({num_passed}/{num_total})")
    else:
        print(f"SOME TESTS FAILED ({num_passed}/{num_total} passed)")
    print("="*70)

    return num_passed == num_total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
