# Submission Instructions

Output format requirements:
- CSV with exactly two columns: `sample_id`, `price`
- `sample_id` must match IDs in `dataset/test.csv` exactly and appear once per row
- `price` must be a positive float

Example header:

sample_id,price

Scoring metric (SMAPE):

SMAPE = (1/n) * Σ |pred - actual| / ((|actual| + |pred|)/2) * 100%

Checklist before submission:
- Ensure number of rows in your output equals number of rows in `dataset/test.csv`
- Save the CSV as `test_out.csv` and upload to the portal
- Include the one-page methodology document `Documentation_template.md` filled by your team
