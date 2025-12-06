# Nigeria Hospital Patients ML Project

End-to-end preprocessing and EDA pipeline for the dataset:

hf://datasets/electricsheepafrica/Nigeria-hospital-patients/patients.parquet

This project loads the dataset directly from Hugging Face using fsspec, reports basic statistics, performs exploratory data analysis with visualizations, and builds a reusable scikit-learn preprocessing pipeline. Artifacts and figures are saved to dedicated folders.

## Dataset structure
- patient_id (object): Unique patient identifier.
- name (object): Patient name.
- age (int64): Age in years; 0 may represent newborns.
- arrival_date (datetime64[ns]): Admission/arrival timestamp.
- departure_date (datetime64[ns]): Discharge/departure timestamp.
- service (object): Clinical service/department (e.g., surgery, emergency, ICU, general_medicine).
- satisfaction (int64): Patient satisfaction score (observed 60–99).

## Features
- Loads parquet from Hugging Face `hf://` URLs via fsspec
- Prints and saves dataset schema, missingness, numeric and categorical summaries
- Visualizations: histograms, boxplots, categorical counts, correlation heatmap, missingness bars, optional time-series counts if a datetime column exists
- Type inference for numeric/categorical/datetime columns (with safe, explainable logic)
- Robust preprocessing pipeline using scikit-learn:
  - Numeric: impute (median) + scale (StandardScaler)
  - Categorical: impute (most_frequent) for missing values + OneHotEncoder (handles unknowns)
  - Datetime: optional extraction of calendar features (year, month, day, dow)
- Pipeline persisted with joblib

## Project Structure
- main.py
- src/
  - preprocessing_utils.py
- reports/
  - figures/eda/
- artifacts/
- data/
  - processed/

## Requirements
- Python 3.10+

Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage
Basic run with defaults (uses the specified HF dataset):

```bash
python main.py \
  --hf-path hf://datasets/electricsheepafrica/Nigeria-hospital-patients/patients.parquet \
  --outdir reports/figures/eda \
  --artifacts artifacts \
  --processed-out data/processed \
  --save-figs True \
  --pairplot False \
  --sample-n 50000
```

Key arguments:
- `--hf-path`: hf:// URL to the parquet file
- `--target`: optional target column name to exclude from features when fitting the preprocessor
- `--datetime-col`: optional explicit name of the primary datetime column
- `--save-figs`: whether to save figures to disk
- `--pairplot`: whether to create a seaborn pairplot (can be slow on large data)
- `--sample-n`: optional row cap for EDA to keep runs fast
- `--max-cats`: maximum distinct categories to display in bar charts

Outputs:
- Figures in `reports/figures`
- Stats CSVs in `reports`
- Preprocessing pipeline `artifacts/preprocessing_pipeline.joblib`
- Optionally, a processed sample parquet in `data/processed`

## Reports: file descriptions
- **schema_dtypes.csv**: Two columns: `column`, `dtype`. Shows each column's pandas dtype at export time.
- **missing_values.csv**: Columns: `column`, `missing_count`, `missing_pct`. `missing_pct` is `(missing_count / total_rows) * 100`.
- **numeric_describe.csv**: Descriptive statistics (pandas `describe()`) for numeric columns only; includes count, mean, std, min, 25%, 50%, 75%, max.
- **categorical_cardinality.csv**: For columns with object/category/bool dtype at export time, lists `column` and `n_unique` (distinct non-null values). Datetime-typed columns are excluded.
- **overview.json**: JSON summary containing `shape`, `memory_mb`, `dtypes` (mapping), `missing_counts` (mapping), `missing_pct` (mapping), and `n_unique_per_col` (mapping).

## Authentication (optional)
Public datasets require no token. For private repos, set an environment variable:

- Windows PowerShell
```powershell
$env:HF_TOKEN = "<your_hf_token>"
```

The loader will automatically use `HF_TOKEN` if available.

## Notes
- Plots use a non-interactive backend and are saved to disk; no GUI is required.
- The pipeline is generic and can be reused across related datasets with similar schema.

## Windsurf Usage

This project was developed with assistance from the Windsurf AI coding assistant. Windsurf was used in the following ways:

- **Pipeline design and refactoring**
  - Helped design a reusable scikit-learn preprocessing pipeline using `ColumnTransformer` and `Pipeline` with separate numeric, categorical, and datetime branches.
  - Suggested separating EDA figures into `reports/figures/eda` and experiment plots into `reports/figures/experiments` for clearer structure.

- **Debugging and error resolution**
  - Assisted in fixing issues such as:
    - `KeyError` / `ValueError` from `ColumnTransformer` when dropped columns were still referenced.
    - Type problems where `age_bucket` was treated as numeric instead of categorical.
    - Adding `get_feature_names_out` to the custom `DatetimeFeaturizer` so transformed feature names could be inspected.
  - Helped reason about why `departure_date` and `satisfaction` are leaky for predicting `service`, leading to their exclusion from the feature set.

- **Feature engineering ideas and documentation**
  - Proposed and refined non-leaky arrival-based features such as `age_bucket`, `arrival_date_month`, `arrival_date_weekday/weekend`, `arrival_dow`, `arrival_part_of_month`, `arrival_quarter`, and `arrival_season`.
  - Helped document the rationale for each engineered feature and its relationship to the modeling target in `reports/potential_issues.md` and `reports/insights.md`.

- **Modeling and interpretation support**
  - Assisted in wiring baseline models (multinomial logistic regression as the **baseline**, and RandomForestClassifier as a non-linear comparator) on top of the shared preprocessor.
  - Helped interpret classification reports and connect the weak performance of both models to limited non-leaky feature signal rather than implementation errors.

- **Prompts that worked well**
  - Targeted debugging prompts (e.g., "Explain this `ValueError` in the preprocessing pipeline and how to fix it").
  - Design prompts (e.g., "Design a reusable preprocessing pipeline with numeric, categorical, and datetime branches").
  - Reporting prompts (e.g., "Summarize these model performance numbers and what they mean").

- **What was modified manually**
  - Final decisions about which columns were treated as leaky or excluded (e.g., `departure_date`, `satisfaction`).
  - Exact definitions of Nigerian seasons and arrival-based buckets.
  - Wording and emphasis in `README.md`, `insights.md`, and `potential_issues.md` to align with the personal writing style

## Insights and Potential Issues

  Links to insights and potential issues reports

- [Insights](reports/insights.md)
- [Potential Issues](reports/potential_issues.md)
