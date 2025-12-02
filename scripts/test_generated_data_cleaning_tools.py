"""Basic runtime tests for generated data cleaning tools.

This script imports the auto-generated functions in
`multi_agents/tools/generated/data_cleaning.py` and runs lightweight
sanity checks to ensure they execute without errors on sample data.

It prints a PASS/FAIL line for each function along with brief diagnostics.

Usage:
    python scripts/test_generated_data_cleaning_tools.py 

Optional flags:
    --use-titanic   Run tests on a subset of the Titanic train.csv data instead of synthetic data.
"""

import os
import sys
import argparse
import traceback
import pandas as pd
from typing import List, Dict, Any

ROOT = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(ROOT, '..'))
sys.path.append(ROOT)


GENERATED_MODULE_PATH = os.path.join(
    ROOT,
    'multi_agents', 'generated_tools'
)
sys.path.append(GENERATED_MODULE_PATH)

try:
    from data_cleaning import (
        fill_missing_values,
        outlier_removal,
        categorical_encoder,
        datetime_parser,
        inconsistent_value_fixer,
        remove_duplicates,
    )
except ImportError as e:
    print(f"ERROR: Could not import generated tools: {e}")
    sys.exit(1)


def _make_synthetic_dataframe() -> pd.DataFrame:
    return pd.DataFrame({
        'num_a': [1, 2, None, 4, 1000],
        'num_b': [10, 10, 10, 10, 10],
        'cat_x': ['Red', 'Blue', 'Blue', 'Green', 'Blue'],
        'cat_y': ['A', 'A', 'B', 'C', 'A'],
        'date_raw': ['2024-01-01', '2024-01-02', '2024/01/03', 'invalid', '2024-01-05'],
        'dup_key': [1,1,2,3,3],
    })


def _load_titanic_subset(root: str) -> pd.DataFrame:
    path = os.path.join(root, 'multi_agents', 'competition', 'titanic', 'train.csv')
    if not os.path.exists(path):
        raise FileNotFoundError(f"Titanic dataset not found at {path}")
    df = pd.read_csv(path)
    # Keep a small subset of columns for speed & clarity
    keep_cols = [c for c in ['Age', 'Fare', 'Cabin', 'Embarked', 'Sex', 'Pclass'] if c in df.columns]
    return df[keep_cols].head(200).copy()


def run_test(name: str, func, *args, **kwargs) -> Dict[str, Any]:
    try:
        result = func(*args, **kwargs)
        return {"name": name, "status": "PASS", "shape": getattr(result, 'shape', None)}
    except Exception as e:
        return {"name": name, "status": "FAIL", "error": str(e), "trace": traceback.format_exc()}


def main():
    parser = argparse.ArgumentParser(description='Test generated data cleaning tools.')
    parser.add_argument('--use-titanic', action='store_true', help='Use Titanic subset instead of synthetic data.')
    args = parser.parse_args()

    if args.use_titanic:
        try:
            df = _load_titanic_subset(ROOT)
            print("Loaded Titanic subset:", df.shape)
        except Exception as e:
            print(f"Failed to load Titanic data: {e}; falling back to synthetic.")
            df = _make_synthetic_dataframe()
    else:
        df = _make_synthetic_dataframe()
        print("Using synthetic dataframe:", df.shape)

    # Ensure categorical/object dtypes for relevant columns
    for col in df.columns:
        if df[col].dtype == 'object' and col.startswith('cat_'):
            df[col] = df[col].astype('category')

    results: List[Dict[str, Any]] = []

    # 1. fill_missing_values
    if 'num_a' in df.columns:
        results.append(run_test('fill_missing_values', fill_missing_values, df.copy(), ['num_a']))
    elif 'Age' in df.columns:
        results.append(run_test('fill_missing_values', fill_missing_values, df.copy(), ['Age']))

    # 2. outlier_removal
    target_cols = [c for c in ['num_a', 'Fare', 'Age'] if c in df.columns]
    if target_cols:
        results.append(run_test('outlier_removal', outlier_removal, df.copy(), target_cols))

    # 3. categorical_encoder
    cat_cols = [c for c in ['cat_x', 'Embarked', 'Sex'] if c in df.columns]
    if cat_cols:
        results.append(run_test('categorical_encoder', categorical_encoder, df.copy(), cat_cols))

    # 4. datetime_parser
    if 'date_raw' in df.columns:
        results.append(run_test('datetime_parser', datetime_parser, df.copy(), ['date_raw']))

    # 5. inconsistent_value_fixer
    if 'cat_y' in df.columns:
        mapping = {'A': 'GroupA', 'B': 'GroupB'}
        results.append(run_test('inconsistent_value_fixer', inconsistent_value_fixer, df.copy(), ['cat_y'], mapping))

    # 6. remove_duplicates
    if 'dup_key' in df.columns:
        results.append(run_test('remove_duplicates', remove_duplicates, df.copy(), subset='dup_key'))

    print("\nRESULTS:")
    for r in results:
        if r['status'] == 'PASS':
            print(f"[PASS] {r['name']} -> shape={r['shape']}")
        else:
            print(f"[FAIL] {r['name']} -> {r['error']}")
            # Uncomment to see tracebacks
            # print(r['trace'])

    # Summary
    passed = sum(1 for r in results if r['status'] == 'PASS')
    total = len(results)
    print(f"\nSummary: {passed}/{total} passed.")
    if passed != total:
        sys.exit(1)


if __name__ == '__main__':
    main()
