import os
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from src.features import get_text_embeddings, get_tabular_features
from src.model import MLPRegressor

def smape(y_true, y_pred):
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0 + 1e-9
    return np.mean(np.abs(y_true - y_pred) / denom) * 100.0

DATASET_FOLDER = 'dataset/'
train = pd.read_csv(os.path.join(DATASET_FOLDER, 'train.csv'))

# Recompute features (may take time; uses sentence-transformers)
print('Computing text embeddings...')
text_emb = get_text_embeddings(train['catalog_content'])
print('Computing tabular features...')
tabular = get_tabular_features(train)
X = np.concatenate([text_emb, tabular], axis=1)
y = np.log1p(train['price'].clip(lower=0.01)).astype(np.float32)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15, random_state=42)

# Ensure numpy arrays
X_val = np.asarray(X_val)
y_val = np.asarray(y_val).astype(np.float32)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device for eval:', device)

model = MLPRegressor(input_dim=X.shape[1])
state_path = 'best_model.pth'
if not os.path.exists(state_path):
    raise FileNotFoundError(f"Model file not found: {state_path}")
model.load_state_dict(torch.load(state_path, map_location='cpu'))
model.to(device)
model.eval()

preds = []
with torch.no_grad():
    batch_size = 256
    for i in range(0, len(X_val), batch_size):
        xb = torch.tensor(X_val[i:i+batch_size], dtype=torch.float32).to(device)
        p = model(xb).cpu().numpy()
        preds.append(p)
preds = np.concatenate(preds)

pred_price = np.expm1(preds)
true_price = np.expm1(y_val)
val_smape = smape(true_price, pred_price)
print(f'Validation SMAPE: {val_smape:.4f}% (n={len(true_price)})')

# Print a few example predictions
df_examples = pd.DataFrame({'true_price': true_price, 'pred_price': pred_price})
print(df_examples.sample(10).to_string(index=False))

print('\nDone.')
