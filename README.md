# Explainable AI for Financial Fraud Detection: Pipeline Notebook

Code for the COMP971 dissertation *Explainable AI for Financial Fraud Detection: A Comparative Study of Deep Learning Models with SHAP and LIME Interpretability* (Ved Bhatkar, Auckland University of Technology, 2026).

The notebook trains three architectures on two public fraud datasets, explains every trained model with SHAP and LIME, and measures how far the two explanation methods agree.

| | |
|---|---|
| Architectures | XGBoost, TabNet, LSTM |
| Datasets | IEEE-CIS Fraud Detection, Bank Account Fraud (BAF) Base variant |
| Explanation methods | SHAP, LIME |
| Agreement measures | Jaccard overlap and Spearman correlation of the top 10 features |
| Notebook | `fraud_detection_pipeline.ipynb` (written for Google Colab) |

## What the notebook does

1. **Loads the data.** IEEE-CIS is two tables (`train_transaction.csv` and `train_identity.csv`), joined on `TransactionID`. BAF is one file (`Base.csv`). Nothing else is changed in the raw data.
2. **Preprocesses both datasets with one pipeline.**
   - Median imputation for numeric columns and a "missing" category for categorical columns.
   - Target-aware encoding for XGBoost, and ordinal encoding with standard scaling for TabNet and LSTM.
   - Class weights in the loss to handle imbalance.
   - A stratified 70/15/15 train, validation and test split with a fixed seed, reused for all three architectures.
3. **Trains six models** (three architectures on two datasets).
   - XGBoost: grid search over depth (4, 6), learning rate (0.05, 0.1) and L2 regularisation (1, 5), with 300 trees and early stopping on validation AUC ROC.
   - TabNet: fixed configuration (`n_d = n_a = 16`, 4 steps, `gamma = 1.5`), early stopping on validation AUC ROC with a patience of 15 and a ceiling of 60 epochs.
   - LSTM: two layers of 64 units, dropout 0.3, 10 epochs, learning rate 0.001, batch size 512. Each transaction is a sequence of length one, because neither dataset has a reliable account key.
4. **Evaluates** F1 score, AUC ROC, precision and recall on the held-out test set at a fixed 0.5 threshold.
5. **Explains the models.**
   - SHAP: TreeSHAP for XGBoost, KernelExplainer (100 background and 100 permutation samples) for TabNet and LSTM, on 200 random test instances per model.
   - LIME: 50 random test instances per model, with absolute weights summed per feature to give one ranking.
   - LIME stability: 20 instances explained 5 times each with different seeds, run once for XGBoost on IEEE-CIS.
6. **Compares explanations.** Jaccard overlap and Spearman correlation between the SHAP and LIME top 10 lists for each model, plus the three-way overlap of SHAP top 10 lists across architectures.

## Results

Test set results at a threshold of 0.5 (from the notebook's saved output):

| Dataset | Model | F1 score | AUC ROC | Precision | Recall |
|---|---|---|---|---|---|
| IEEE-CIS | XGBoost | 0.434 | 0.945 | 0.295 | 0.821 |
| IEEE-CIS | TabNet | 0.341 | 0.925 | 0.216 | 0.809 |
| IEEE-CIS | LSTM | 0.263 | 0.851 | 0.163 | 0.671 |
| BAF | XGBoost | 0.101 | 0.903 | 0.054 | 0.799 |
| BAF | TabNet | 0.081 | 0.890 | 0.043 | 0.831 |
| BAF | LSTM | 0.084 | 0.889 | 0.044 | 0.826 |

The dissertation discusses these results, the SHAP and LIME agreement scores and their limits in full.

## Data

The datasets are not included in this repository. Download them and set the paths in the Config cell.

- **IEEE-CIS:** https://www.kaggle.com/competitions/ieee-fraud-detection/data. Use `train_transaction.csv` and `train_identity.csv`. The test files have no fraud labels and are not used.
- **BAF (NeurIPS 2022):** https://www.kaggle.com/datasets/sgpjesus/bank-account-fraud-dataset-neurips-2022. Use `Base.csv`.

## How to run

1. Open the notebook in Google Colab and select a GPU runtime (Runtime, then Change runtime type).
2. Upload the three CSV files to Google Drive.
3. Edit the paths and column names in the Config cell (`IEEE_TRANSACTION_PATH`, `IEEE_IDENTITY_PATH`, `BAF_PATH`).
4. Run all cells in order. The setup cell installs the dependencies:

```
pip install pytorch-tabnet xgboost shap lime imbalanced-learn scikit-learn matplotlib-venn
```

Preprocessing IEEE-CIS needs roughly 8 to 12 GB of RAM. Colab's default memory is enough. KernelExplainer and LIME are the slowest steps.

## Notes and limitations

- Each model was trained once with one seed and one split, so the results are descriptive and carry no confidence intervals.
- Only XGBoost is tuned by grid search. TabNet and LSTM use fixed configurations.
- XGBoost is explained by TreeSHAP and the other two models by KernelExplainer, so part of any difference in agreement may come from the explainer.
- SMOTE is defined in the notebook as an optional secondary comparison but is not used in any reported result.
- Only the Base variant of BAF is used. The other five variants are not evaluated.
- Explanations are computed on random samples, which contain few fraud cases.

## Citation

Bhatkar, V. (2026). *Explainable AI for financial fraud detection: A comparative study of deep learning models with SHAP and LIME interpretability* [Master's dissertation]. Auckland University of Technology.

## Contact

Ved Bhatkar, https://github.com/SeasonS2234
