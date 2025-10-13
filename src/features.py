import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

def get_text_embeddings(series):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    return model.encode(series.tolist(), batch_size=256, show_progress_bar=True)

def extract_ipq(text):
    import re
    m = re.search(r'IPQ\s*([0-9]+)', text)
    return int(m.group(1)) if m else 1

def get_tabular_features(df):
    df['IPQ'] = df['catalog_content'].apply(extract_ipq)
    return df[['IPQ']].to_numpy()

def get_image_embeddings(image_paths):
    import torch
    import timm
    from PIL import Image
    from torchvision import transforms

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = timm.create_model('efficientnet_b0', pretrained=True, num_classes=0)
    model.eval()
    model.to(device)

    transf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    feats = []
    for img_path in image_paths:
        try:
            img = Image.open(img_path).convert("RGB")
            img = transf(img).unsqueeze(0).to(device)
            with torch.no_grad():
                embed = model(img).cpu().squeeze().numpy()
        except:
            embed = np.zeros(1280)
        feats.append(embed)
    return np.stack(feats)
