import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from src.features import get_text_embeddings, get_tabular_features, get_image_embeddings # add image if needed
from src.model import MLPRegressor
import numpy as np
import json
import platform

def smape(y_true, y_pred):
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0 + 1e-9
    return np.mean(np.abs(y_true - y_pred) / denom) * 100.0

DATASET_FOLDER = 'dataset/'
train = pd.read_csv(os.path.join(DATASET_FOLDER, 'train.csv'))

# Optional debug mode: limit number of rows and/or epochs via env vars
DEBUG_SUBSET = int(os.getenv('DEBUG_SUBSET', '0'))
DEBUG_EPOCHS = int(os.getenv('DEBUG_EPOCHS', '0'))
if DEBUG_SUBSET > 0:
    train = train.head(DEBUG_SUBSET)

# --- Feature extraction ---
text_emb = get_text_embeddings(train['catalog_content'])
tabular = get_tabular_features(train)

# Optional image embeddings
USE_IMAGE = int(os.getenv('USE_IMAGE', '0'))
image_emb = None
if USE_IMAGE:
    # cache image embeddings to avoid repeated downloads
    image_cache = os.path.join(DATASET_FOLDER, 'image_emb.npy')
    if os.path.exists(image_cache):
        print('Loading cached image embeddings...')
        image_emb = np.load(image_cache)
    else:
        print('Computing image embeddings (this may take time)...')
        # The dataset must have an 'image_link' column; download images first if needed
        test_image_folder = os.path.join(DATASET_FOLDER, 'images')
        from src.utils import download_images
        download_images(train['image_link'].tolist(), test_image_folder)
        img_paths = [os.path.join(test_image_folder, os.path.basename(x)) for x in train['image_link'].fillna('')]
        image_emb = get_image_embeddings(img_paths)
        np.save(image_cache, image_emb)

if image_emb is not None:
    X = np.concatenate([text_emb, tabular, image_emb], axis=1)
else:
    X = np.concatenate([text_emb, tabular], axis=1)
y = np.log1p(train['price'].clip(lower=0.01)).astype(np.float32)  # log1p target

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15, random_state=42)

# Ensure arrays (not pandas Series) for safe tensor conversion
X_train = np.asarray(X_train)
X_val = np.asarray(X_val)
y_train = np.asarray(y_train).astype(np.float32)
y_val = np.asarray(y_val).astype(np.float32)

# Device and model
cuda_available = torch.cuda.is_available()
device = torch.device('cuda' if cuda_available else 'cpu')
print(f"Using device: {device}")
if cuda_available:
    print(f"CUDA device count: {torch.cuda.device_count()}, name: {torch.cuda.get_device_name(0)}")

model = MLPRegressor(input_dim=X.shape[1]).to(device)
loss_fn = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-3)
EPOCHS = 35
if DEBUG_EPOCHS > 0:
    EPOCHS = DEBUG_EPOCHS

# For live plots
history = {'loss': [], 'val_loss': [], 'val_smape': []}

# Early stopping settings
best_smape = 1e9
epochs_no_improve = 0
PATIENCE = 7  # stop after 7 epochs of no val SMAPE improvement

# Create DataLoaders that keep data on CPU and stream batches to GPU to avoid OOM
# Choose a larger batch size when CUDA is available
default_batch = 256 if cuda_available else 128
batch_size = int(os.getenv('BATCH_SIZE', str(default_batch)))
# Convert to tensors on CPU (do NOT .to(device) here)
X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32)
X_val_t = torch.tensor(X_val, dtype=torch.float32)
y_val_t = torch.tensor(y_val, dtype=torch.float32)

pin_memory = True if cuda_available else False
# Tune num_workers: 0 on Windows to avoid multiprocessing issues, else allow small parallelism
default_workers = 0 if platform.system().lower().startswith('win') else 4
num_workers = int(os.getenv('NUM_WORKERS', str(default_workers)))
train_ds = TensorDataset(X_train_t, y_train_t)
val_ds = TensorDataset(X_val_t, y_val_t)
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

# Setup AMP scaler if CUDA available
scaler = torch.cuda.amp.GradScaler() if cuda_available else None

# Save training config for reproducibility
train_config = {
    'epochs': EPOCHS,
    'batch_size': batch_size,
    'num_workers': num_workers,
    'device': str(device),
    'cuda_available': bool(cuda_available),
}
with open('train_config.json', 'w') as f:
    json.dump(train_config, f, indent=2)
print('Training config saved to train_config.json')

# Ensemble support: train ENSEMBLE_SIZE models with different seeds
ENSEMBLE_SIZE = int(os.getenv('ENSEMBLE_SIZE', '1'))
BASE_SEED = int(os.getenv('BASE_SEED', '42'))

all_histories = []
for model_idx in range(ENSEMBLE_SIZE):
    seed = BASE_SEED + model_idx
    torch.manual_seed(seed)
    np.random.seed(seed)
    print(f"Starting training for model {model_idx} with seed {seed}")

    # recreate model/optimizer for each ensemble member
    model = MLPRegressor(input_dim=X.shape[1]).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)

    # per-model training state
    history = {'loss': [], 'val_loss': [], 'val_smape': []}
    best_smape = 1e9
    epochs_no_improve = 0

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        for xb_cpu, yb_cpu in train_loader:
            xb = xb_cpu.to(device, non_blocking=True)
            yb = yb_cpu.to(device, non_blocking=True)
            optimizer.zero_grad()
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    preds = model(xb)
                    loss = loss_fn(preds, yb)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                preds = model(xb)
                loss = loss_fn(preds, yb)
                loss.backward()
                optimizer.step()
            running_loss = loss.item()
        model.eval()
        val_preds = []
        with torch.no_grad():
            for xb_cpu, _ in val_loader:
                xb = xb_cpu.to(device, non_blocking=True)
                preds = model(xb).cpu().numpy()
                val_preds.append(preds)
        val_preds = np.concatenate(val_preds)
        val_loss = np.mean((val_preds - y_val) ** 2)
        val_price = np.expm1(val_preds)
        val_true_price = np.expm1(y_val)
        val_smape = smape(val_true_price, val_price)
        history['loss'].append(running_loss)
        history['val_loss'].append(val_loss)
        history['val_smape'].append(val_smape)
        print(f"Model {model_idx} Epoch {epoch+1} | Train Loss: {loss.item():.4f} | Val Loss: {val_loss:.4f} | SMAPE: {val_smape:.2f}%")

        if val_smape < best_smape:
            best_smape = val_smape
            epochs_no_improve = 0
            model_file = f'best_model_{model_idx}.pth' if ENSEMBLE_SIZE>1 else 'best_model.pth'
            torch.save(model.state_dict(), model_file)
            print(f"New best model saved at epoch {epoch+1} -> {model_file}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print(f"Early stopping at epoch {epoch+1} for model {model_idx}. Best SMAPE: {best_smape:.2f}%")
                break

    all_histories.append(history)

    # small delay between models
    torch.cuda.empty_cache()

# If ensemble size is 1 we saved best_model.pth, else saved best_model_0..N-1.pth

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    for xb_cpu, yb_cpu in train_loader:
        # Move only the current batch to device (non_blocking if pinned)
        xb = xb_cpu.to(device, non_blocking=True)
        yb = yb_cpu.to(device, non_blocking=True)
        optimizer.zero_grad()
        preds = model(xb)
        loss = loss_fn(preds, yb)
        loss.backward()
        optimizer.step()
        running_loss = loss.item()
    model.eval()
    # Validation: stream batches from CPU to GPU
    val_preds = []
    with torch.no_grad():
        for xb_cpu, _ in val_loader:
            xb = xb_cpu.to(device, non_blocking=True)
            preds = model(xb).cpu().numpy()
            val_preds.append(preds)
    val_preds = np.concatenate(val_preds)
    val_loss = np.mean((val_preds - y_val) ** 2)
    val_price = np.expm1(val_preds)
    val_true_price = np.expm1(y_val)
    val_smape = smape(val_true_price, val_price)
    history['loss'].append(running_loss)
    history['val_loss'].append(val_loss)
    history['val_smape'].append(val_smape)
    print(f"Epoch {epoch+1} | Train Loss: {loss.item():.4f} | Val Loss: {val_loss:.4f} | SMAPE: {val_smape:.2f}%")

    # Early stopping: save best model and stop when no improvement
    if val_smape < best_smape:
        best_smape = val_smape
        epochs_no_improve = 0
        torch.save(model.state_dict(), 'best_model.pth')
        print("New best model saved at epoch", epoch+1)
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= PATIENCE:
            print(f"Early stopping at epoch {epoch+1}. Best SMAPE: {best_smape:.2f}%")
            break

# --- Plot training curves ---
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(history['loss'], label='Train loss')
plt.plot(history['val_loss'], label='Val loss')
plt.legend(), plt.title("Loss")
plt.subplot(1,2,2)
plt.plot(history['val_smape'], label='Val SMAPE')
plt.legend(), plt.title("Validation SMAPE (%)")
plt.savefig('training_curves.png')
plt.show()

