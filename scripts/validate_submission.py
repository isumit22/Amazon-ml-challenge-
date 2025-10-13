"""
Validate a submission CSV has correct format and basic sanity checks.

Usage:
  python scripts/validate_submission.py --submission dataset/test_out.csv --test dataset/test.csv [--truth dataset/train.csv]

Checks performed:
- Header equals `sample_id,price`
- Row count equals test row count
- sample_id uniqueness
- All prices positive floats
- Optional: compute SMAPE if --truth provided (truth CSV must contain sample_id,price)

"""
import argparse
import csv
import sys
from math import isfinite


def read_csv_as_dict(path):
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    return rows


def smape(preds, trues):
    n = 0
    s = 0.0
    for k in preds:
        if k not in trues:
            continue
        p = float(preds[k])
        t = float(trues[k])
        denom = (abs(t) + abs(p)) / 2.0
        if denom == 0:
            continue
        s += abs(p - t) / denom
        n += 1
    return (s / n) * 100 if n else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--submission', required=True)
    parser.add_argument('--test', required=True)
    parser.add_argument('--truth', required=False)
    args = parser.parse_args()

    # Read test file to get sample ids
    test_rows = read_csv_as_dict(args.test)
    test_ids = [r['sample_id'] for r in test_rows]

    sub_rows = read_csv_as_dict(args.submission)

    # Header check
    with open(args.submission, newline='', encoding='utf-8') as f:
        header = f.readline().strip()
    if header != 'sample_id,price':
        print(f'ERROR: submission header is "{header}"; expected "sample_id,price"')
        sys.exit(2)

    # Row count
    if len(sub_rows) != len(test_rows):
        print(f'ERROR: submission has {len(sub_rows)} rows but test has {len(test_rows)} rows')
        sys.exit(2)

    # IDs and uniqueness
    sub_ids = [r['sample_id'] for r in sub_rows]
    if len(set(sub_ids)) != len(sub_ids):
        print('ERROR: duplicate sample_id found in submission')
        sys.exit(2)
    # Check IDs match (set equality)
    if set(sub_ids) != set(test_ids):
        print('ERROR: submission sample_ids do not match test sample_ids exactly')
        # optional detailed report
        extra = set(sub_ids) - set(test_ids)
        missing = set(test_ids) - set(sub_ids)
        print(f'  extra in submission: {len(extra)}; missing from submission: {len(missing)}')
        sys.exit(2)

    # Price checks
    for r in sub_rows:
        try:
            p = float(r['price'])
            if not isfinite(p) or p <= 0:
                print(f'ERROR: non-positive or non-finite price for sample_id {r["sample_id"]}: {r["price"]}')
                sys.exit(2)
        except Exception:
            print(f'ERROR: price not a float for sample_id {r["sample_id"]}: {r["price"]}')
            sys.exit(2)

    print('Submission format checks passed')

    if args.truth:
        truth_rows = read_csv_as_dict(args.truth)
        truth_map = {r['sample_id']: r['price'] for r in truth_rows}
        pred_map = {r['sample_id']: r['price'] for r in sub_rows}
        score = smape(pred_map, truth_map)
        if score is not None:
            print(f'SMAPE on provided truth: {score:.4f}%')
        else:
            print('SMAPE: unable to compute (no overlapping ids)')


if __name__ == '__main__':
    main()
