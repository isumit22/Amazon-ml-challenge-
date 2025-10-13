"""Generate a submission-like CSV using a simple heuristic (median train price).
This is a small fallback to produce a correctly formatted `dataset/test_out.csv`
when a trained model (`best_model.pth`) isn't available or dependencies are heavy.

Usage:
    python -m src.generate_dummy_predictions

Output:
    dataset/test_out.csv
"""
import os
import pandas as pd

DATASET_FOLDER = os.path.join(os.path.dirname(__file__), '..', 'dataset')
DATASET_FOLDER = os.path.normpath(DATASET_FOLDER) + os.sep

train_path = os.path.join(DATASET_FOLDER, 'train.csv')
test_path = os.path.join(DATASET_FOLDER, 'test.csv')
out_path = os.path.join(DATASET_FOLDER, 'test_out.csv')

if not os.path.exists(train_path) or not os.path.exists(test_path):
    raise FileNotFoundError('Make sure dataset/train.csv and dataset/test.csv exist relative to project root')

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

# compute median price (fallback). Ensure positive float and at least 0.01
median_price = float(train['price'].clip(lower=0.01).median())
if median_price <= 0:
    median_price = 1.0

# Create predictions: use median price for all items
preds = [median_price] * len(test)

out = pd.DataFrame({'sample_id': test['sample_id'], 'price': preds})
out.to_csv(out_path, index=False)
print(f'Wrote {out_path} with {len(out)} predictions using median price = {median_price:.4f}')
