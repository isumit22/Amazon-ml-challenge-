
# ML Challenge 2025: Product Price Prediction

## Overview
Predict product prices using text and image features. The solution uses PyTorch, sentence-transformers, and timm for feature extraction and model training. Ensemble methods and image embeddings are supported for improved accuracy.

## Folder Structure
- `src/` — Source code (model, training, prediction, features, utils)
- `dataset/` — Data files (`train.csv`, `test.csv`, etc.)
- `requirements.txt` — Python dependencies
- `best_model_0.pth`, `best_model_1.pth`, `best_model_2.pth` — Ensemble model checkpoints

## Setup Instructions
1. **Install Miniconda** (recommended for CUDA compatibility)
2. **Create environment:**
   ```powershell
   conda create -n pricing python=3.11 -y
   conda activate pricing
   conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
   pip install -r requirements.txt
   ```
3. **Download data:** Place `train.csv` and `test.csv` in `dataset/`.

## Training
Run ensemble training (recommended):
```powershell
python src/train.py --ensemble 3 --base-seed 100
```
This will save `best_model_0.pth`, `best_model_1.pth`, `best_model_2.pth`.

## Prediction
Generate predictions for leaderboard submission:
```powershell
python src/predict.py --ensemble 3
```
Output: `dataset/test_out.csv` (ready for submission)

## Evaluation
To evaluate on validation set:
```powershell
python src/evaluate.py
```

## Notes
- GPU is recommended for training (CUDA-enabled PyTorch required).
- Image embeddings are cached for efficiency.
- For best results, tune model architecture and ensemble size.

## Contact
For issues, contact the repository maintainer.
2. **dataset/test.csv:** Test file without output labels (`price`). Generate predictions using your model/solution on this file's data and format the output file to match sample_test_out.csv
3. **dataset/sample_test.csv:** Sample test input file.
4. **dataset/sample_test_out.csv:** Sample outputs for sample_test.csv. The output for test.csv must be formatted in the exact same way. Note: The predictions in the file might not be correct

### Constraints:

1. You will be provided with a sample output file. Format your output to match the sample output file exactly. 

2. Predicted prices must be positive float values.

3. Final model should be a MIT/Apache 2.0 License model and up to 8 Billion parameters.

### Evaluation Criteria:

Submissions are evaluated using **Symmetric Mean Absolute Percentage Error (SMAPE)**: A statistical measure that expresses the relative difference between predicted and actual values as a percentage, while treating positive and negative errors equally.

**Formula:**
```
SMAPE = (1/n) * Σ |predicted_price - actual_price| / ((|actual_price| + |predicted_price|)/2)
```

**Example:** If actual price = $100 and predicted price = $120  
SMAPE = |100-120| / ((|100| + |120|)/2) * 100% = 18.18%

**Note:** SMAPE is bounded between 0% and 200%. Lower values indicate better performance.

### Leaderboard Information:

- **Public Leaderboard:** During the challenge, rankings will be based on 25K samples from the test set to provide real-time feedback on your model's performance.
- **Final Rankings:** The final decision will be based on performance on the complete 75K test set along with provided documentation of the proposed approach by the teams.

### Submission Requirements:

1. Upload a `test_out.csv` file in the Portal with the exact same formatting as `sample_test_out.csv`

2. All participating teams must also provide a 1-page document describing:
   - Methodology used
   - Model architecture/algorithms selected
   - Feature engineering techniques applied
   - Any other relevant information about the approach
   Note: A sample template for this documentation is provided in Documentation_template.md

### **Academic Integrity and Fair Play:**

**⚠️ STRICTLY PROHIBITED: External Price Lookup**

Participants are **STRICTLY NOT ALLOWED** to obtain prices from the internet, external databases, or any sources outside the provided dataset. This includes but is not limited to:
- Web scraping product prices from e-commerce websites
- Using APIs to fetch current market prices
- Manual price lookup from online sources
- Using any external pricing databases or services

**Enforcement:**
- All submitted approaches, methodologies, and code pipelines will be thoroughly reviewed and verified
- Any evidence of external price lookup or data augmentation from internet sources will result in **immediate disqualification**

**Fair Play:** This challenge is designed to test your machine learning and data science skills using only the provided training data. External price lookup defeats the purpose of the challenge.


### Tips for Success:

- Consider both textual features (catalog_content) and visual features (product images)
- Explore feature engineering techniques for text and image data
- Consider ensemble methods combining different model types
- Pay attention to outliers and data preprocessing
