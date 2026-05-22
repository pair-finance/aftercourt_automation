# Binary Classification: Vermögensverzeichnis Detection

## Objective

Classify documents as **vermögensverzeichnis** (Target 1) or **not vermögensverzeichnis** (Target 0) using TF-IDF features extracted from document text.

## Data

- **Source:** `data/raw/final_raw_data.csv`
- **Target column:** `document_type` (binary-mapped: `vermögensverzeichnis` → 1, all others → 0)
- **Text column:** `cleaned_text` (falls back to `text` when missing)
- **Total samples:** 5,532
- **Class distribution:**

| Class | Count | Percentage |
|---|---|---|
| NOT vermögensverzeichnis | 5,443 | 98.4% |
| vermögensverzeichnis | 89 | 1.6% |

> The dataset is highly imbalanced. All models use `class_weight="balanced"` to compensate.

## Train / Test Split

- **Split ratio:** 80% train / 20% test (stratified)
- **Train:** 4,425 samples (71 positive)
- **Test:** 1,107 samples (18 positive)

## Feature Extraction

**TF-IDF Vectorizer** with the following configuration:

| Parameter | Value |
|---|---|
| `max_features` | 10,000 |
| `ngram_range` | (1, 2) — unigrams + bigrams |
| `min_df` | 2 |
| `max_df` | 0.95 |
| `sublinear_tf` | True |

## Models Evaluated

### 1. Logistic Regression
- `class_weight="balanced"`, `C=1.0`, `max_iter=1000`

### 2. Random Forest
- `n_estimators=200`, `class_weight="balanced"`

### 3. Linear SVC
- `class_weight="balanced"`, `max_iter=2000`

## Results

### Cross-Validation (5-Fold Stratified, F1 Score)

| Model | CV F1 (mean ± std) |
|---|---|
| Logistic Regression | 0.9931 ± 0.0138 |
| Random Forest | 0.9931 ± 0.0138 |
| Linear SVC | 0.9931 ± 0.0138 |

### Test Set Performance

All three models achieved **perfect classification** on the test set:

| Metric | NOT vermögensverz. | vermögensverz. |
|---|---|---|
| Precision | 1.00 | 1.00 |
| Recall | 1.00 | 1.00 |
| F1-score | 1.00 | 1.00 |
| Support | 1,089 | 18 |

- **Accuracy:** 1.00
- **ROC-AUC:** 1.0000 (all models)

## Top Discriminating TF-IDF Features

### Top 15 Keywords **FOR** vermögensverzeichnis (positive coefficient)

| Feature | Coefficient |
|---|---|
| nein | 1.3178 |
| sachen | 0.8057 |
| ja | 0.8017 |
| gegenstände | 0.7932 |
| ja und | 0.7773 |
| und zwar | 0.7650 |
| usw | 0.7571 |
| ehegatten | 0.6973 |
| vermögensverzeichnis | 0.6881 |
| kinder | 0.6833 |
| forderungen | 0.6706 |
| vermögensverzeichnisses | 0.6511 |
| des vermögensverzeichnisses | 0.6423 |
| fahrzeuge | 0.6338 |
| haben sie | 0.6213 |

### Top 15 Keywords **AGAINST** vermögensverzeichnis (negative coefficient)

| Feature | Coefficient |
|---|---|
| 2025 | −0.3871 |
| uhr | −0.2990 |
| der | −0.2909 |
| pair | −0.2864 |
| gmbh | −0.2858 |
| pair finance | −0.2764 |
| berlin | −0.2752 |
| finance gmbh | −0.2460 |
| termin zur | −0.2347 |
| amtsgericht | −0.2303 |
| de | −0.2242 |
| bestimmt | −0.2229 |
| 10623 | −0.2226 |
| 10623 berlin | −0.2210 |
| die | −0.2140 |

## Plots

| Plot | File |
|---|---|
| Confusion Matrix | `assets/data_analysis_plots/vermogensverzeichnis_confusion_matrix.png` |
| ROC Curve | `assets/data_analysis_plots/vermogensverzeichnis_roc_curve.png` |
| Top Features | `assets/data_analysis_plots/vermogensverzeichnis_top_features.png` |

### Confusion Matrix

![Confusion Matrix](../../assets/data_analysis_plots/vermogensverzeichnis_confusion_matrix.png)

### ROC Curve

![ROC Curve](../../assets/data_analysis_plots/vermogensverzeichnis_roc_curve.png)

### Top Features

![Top Features](../../assets/data_analysis_plots/vermogensverzeichnis_top_features.png)

## Key Takeaways

1. **Vermögensverzeichnis documents are highly separable** from other document types using simple TF-IDF + linear models — all models achieve perfect test performance.
2. **Domain-specific vocabulary** like *nein*, *ja*, *sachen*, *gegenstände*, *ehegatten*, *kinder*, *fahrzeuge* strongly signals this document type (asset declaration questionnaire format).
3. **Negative features** are generic legal/company terms (*pair finance*, *amtsgericht*, *berlin*, *uhr*) common across other document types.
4. Given the perfect separability, a **Logistic Regression with TF-IDF** is sufficient — no need for heavier models.

## How to Run

```bash
python notebooks/classification/vermogensverzeichnis_binary_classifier.py
```

## Dependencies

- pandas, numpy
- scikit-learn
- matplotlib
