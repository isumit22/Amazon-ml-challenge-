import os
import pandas as pd
import torch
import numpy as np
from src.features import get_text_embeddings, get_tabular_features, get_image_embeddings
from src.model import MLPRegressor
import os
import numpy as np

DATASET_FOLDER = 'dataset/'
test = pd.read_csv(os.path.join(DATASET_FOLDER, 'test.csv'))


# Features
text_emb = get_text_embeddings(test['catalog_content'])
tabular = get_tabular_features(test)
USE_IMAGE = int(os.getenv('USE_IMAGE', '0'))
image_emb = None
if USE_IMAGE:
    image_cache = os.path.join('dataset', 'image_emb.npy')
    if os.path.exists(image_cache):
        image_emb = np.load(image_cache)
    else:
        print('Image embeddings not found; skipping image features')

if image_emb is not None:
    X = np.concatenate([text_emb, tabular, image_emb], axis=1)
else:
    X = np.concatenate([text_emb, tabular], axis=1)

ENSEMBLE_SIZE = int(os.getenv('ENSEMBLE_SIZE', '1'))
all_preds = []
for i_model in range(ENSEMBLE_SIZE):
    model = MLPRegressor(input_dim=X.shape[1])
    model_file = f'best_model.pth' if ENSEMBLE_SIZE == 1 else f'best_model_{i_model}.pth'
    model.load_state_dict(torch.load(model_file, map_location='cpu'))
    model.eval()
    with torch.no_grad():
        preds = []
        for i in range(0, len(X), 128):
            batch = torch.tensor(X[i:i+128], dtype=torch.float32)
            pb = model(batch).numpy()
            preds.append(pb)
        preds = np.concatenate(preds)
    all_preds.append(preds)

preds = np.mean(np.stack(all_preds, axis=0), axis=0)
price_preds = np.maximum(np.expm1(preds), 0.01)

out = pd.DataFrame({'sample_id': test['sample_id'], 'price': price_preds})
out.to_csv(os.path.join(DATASET_FOLDER, 'test_out.csv'), index=False)
print("Predictions saved to dataset/test_out.csv")
