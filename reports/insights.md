# Service-focused insights

This report is auto-generated from the latest EDA run. It summarizes how the `service` field relates to patient age and satisfaction.

## Service distribution

- **emergency**: 263 patients (~26.3% of sample)
- **surgery**: 254 patients (~25.4% of sample)
- **general_medicine**: 242 patients (~24.2% of sample)
- **ICU**: 241 patients (~24.1% of sample)

Associated figure: `service_distribution_counts.png` in `reports/figures/`.

## Age by service

- **ICU**: age mean=44.2, std=25.9, min=0, 25%=23, median=43, 75%=68, max=88
- **emergency**: age mean=45.3, std=26.3, min=0, 25%=24, median=47, 75%=69, max=89
- **general_medicine**: age mean=46.2, std=24.7, min=0, 25%=26, median=50, 75%=66, max=89
- **surgery**: age mean=45.6, std=27.0, min=0, 25%=20, median=45, 75%=69, max=89

Associated figure: `service_age_boxplot.png` in `reports/figures/`.

## Satisfaction by service

- **ICU**: satisfaction mean=79.9, std=11.7, min=60, 25%=69, median=81, 75%=90, max=99
- **emergency**: satisfaction mean=79.5, std=11.6, min=60, 25%=70, median=80, 75%=89, max=99
- **general_medicine**: satisfaction mean=78.6, std=11.1, min=60, 25%=69, median=78, 75%=88, max=99
- **surgery**: satisfaction mean=80.3, std=11.7, min=60, 25%=71, median=80, 75%=91, max=99

Associated figure: `service_satisfaction_boxplot.png` in `reports/figures/`.

## Correlation of numeric features with `service`

- **age vs service (encoded)**: Pearson r = 0.021027. This is very close to 0, suggesting no meaningful linear relationship between this numeric feature and the encoded service labels.
- **satisfaction vs service (encoded)**: Pearson r = 0.003044. This is very close to 0, suggesting no meaningful linear relationship between this numeric feature and the encoded service labels.

Associated figure: `service_numeric_correlations_heatmap.png` in `reports/figures/`.

## Model comparison: LogisticRegression vs RandomForestClassifier

Two models were trained to predict `service` using the same non-leaky feature set and preprocessing pipeline. Multinomial logistic regression is treated as the **baseline** model, with RandomForestClassifier providing a more flexible non-linear comparison:

- **Multinomial logistic regression (baseline)**
  - Test accuracy: ~0.24 (24%).
  - Macro-averaged F1: ~0.24.
  - Per-class F1-scores are in the ~0.17–0.28 range, indicating weak but non-zero signal.

- **RandomForestClassifier**
  - Test accuracy: ~0.26 (26%).
  - Macro-averaged F1: ~0.26.
  - Per-class F1-scores are in the ~0.20–0.32 range, slightly better than logistic regression but still modest.

With four services of roughly similar prevalence, random guessing would yield accuracy around 25%. Both models perform only slightly above this baseline, confirming that the currently allowed non-leaky features (age and simple arrival-time/seasonality signals) contain limited discriminative information about `service`. The main limitation is feature signal rather than model choice; substantial performance gains would likely require richer pre-admission variables (e.g., clinical status, comorbidities, triage information) that remain available at the time of service assignment.

## Conclusion: feature set evolution and impact on models

The initial modeling experiments used a very small feature set drawn directly from the raw schema (essentially age and arrival/departure dates, with minimal transformation). After iterating on the preprocessing pipeline, the final feature set now includes:

- Core numeric inputs (e.g., `age`, `arrival_date_month`).
- Multiple non-leaky, arrival-based temporal features (`arrival_date_weekday/weekend`, `arrival_dow`, `arrival_part_of_month`, `arrival_quarter`, `arrival_season`).
- Bucketed age groups (`age_bucket`).

This expansion increased the total number of engineered input features (and, after one-hot encoding, the dimensionality seen by the models) and led to modest improvements in performance:

- Multinomial logistic regression moved from ~0.22 to ~0.24 accuracy.
- RandomForestClassifier achieved ~0.26 accuracy and slightly higher macro F1.

However, even with the richer feature space, both models remain only marginally better than a random 4-class guess. This suggests that, under the constraint of using only non-leaky, admission-time information present in this dataset, there is limited predictive signal for which service a patient ultimately receives. Further gains are unlikely to come from additional re-encoding of the same variables; instead, they would require new, clinically meaningful pre-admission features (e.g., vitals, presenting complaint, comorbidity burden, or facility-level context) that are not currently available in the dataset.

