# Ames House Price Prediction

End-to-end regression project on the Ames Housing dataset, including:

- Exploratory Data Analysis (EDA)
- Feature engineering and preprocessing
- Multiple tree-based models (Random Forest, Gradient Boosting, optionally LightGBM/CatBoost)
- Model evaluation and explainability (permutation importance, SHAP)
- A simple Streamlit web app to predict house prices

---

## 1. Project Structure

Key files in this folder:

- `AmesHousing.csv` – raw dataset.
- `ames-house-prediction.ipynb` – main notebook with EDA, preprocessing, modeling, tuning, and explainability.
- `best_ames_model_fe.joblib` – trained pipeline (preprocessing + best model) saved for deployment.
- `app.py` – Streamlit app that loads `best_ames_model_fe.joblib` and serves predictions.
- `requirements.txt` – Python dependencies for this project.

---

## 2. Environment Setup

You can run this project either locally (recommended for the Streamlit app) or in Google Colab (for the notebook).

### 2.1 Local Setup (Python 3.9+ recommended)

From the repo root (or `House Prediction` folder):

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

This installs all libraries needed for the notebook, model training, SHAP, and the Streamlit app.

---

## 3. Running the Notebook

You can explore and extend the analysis via Jupyter or VS Code notebooks.

### 3.1 Launch locally

From the `House Prediction` directory:

```bash
cd "House Prediction"
jupyter notebook  # or: jupyter lab
```

Then open `ames-house-prediction.ipynb` and run the cells from top to bottom. The notebook will:

- Load `AmesHousing.csv`.
- Perform extensive EDA on numeric and categorical features.
- Engineer additional features (e.g., house age, total bathrooms, total square footage).
- Apply preprocessing (imputation, scaling, `OrdinalEncoder`, `OneHotEncoder`).
- Train multiple models and perform hyperparameter tuning (e.g., with `RandomizedSearchCV`).
- Evaluate models on a hold-out test set with RMSE and R².
- Generate explainability plots: permutation importance and SHAP summaries.

### 3.2 Running in Google Colab

If you prefer Colab:

1. Upload `AmesHousing.csv` and copy the notebook code cells into a new Colab notebook.
2. In the first cell, install extra packages, for example:

```python
!pip install shap xgboost lightgbm catboost seaborn matplotlib scikit-learn
```

3. Adjust the dataset path if necessary (e.g., `csv_path = "AmesHousing.csv"`).

---

## 4. Using the Trained Model Programmatically

The file `best_ames_model_fe.joblib` is a serialized **scikit-learn pipeline** that includes:

- All preprocessing steps (`ColumnTransformer` with numeric scaling, ordinal + one-hot encoders, imputers).
- The best-performing model (e.g., tuned Random Forest, LightGBM, or CatBoost, depending on your training run).

Example usage in Python:

```python
import joblib
import pandas as pd
import numpy as np

model = joblib.load("best_ames_model_fe.joblib")

sample = pd.DataFrame([
		{
				"GrLivArea": 1500,
				"TotalBsmtSF": 800,
				"YearBuilt": 1990,
				"YearRemodAdd": 1995,
				"FullBath": 2,
				"HalfBath": 1,
				"BsmtFullBath": 0,
				"BsmtHalfBath": 0,
				"YrSold": 2010,
				"Neighborhood": "NAmes",
				"HouseStyle": "1Story",
				"OverallQual": 5,
				"ExterQual": "TA",
				"KitchenQual": "TA",
				# other columns can be omitted and will be imputed as NaN
		}
])

pred_log = model.predict(sample)[0]
pred_price = np.expm1(pred_log)
print(f"Predicted Sale Price: ${pred_price:,.0f}")
```

The pipeline will impute missing columns as long as the provided ones use the original training column names.

---

## 5. Running the Streamlit App

The Streamlit app (`app.py`) provides a simple UI to get price predictions.

### 5.1 Start the app

From the `House Prediction` folder:

```bash
cd "House Prediction"
streamlit run app.py
```

Streamlit will print a local URL (e.g., `http://localhost:8501`). Open it in your browser.

### 5.2 How the app works

- The app loads `best_ames_model_fe.joblib`.
- The sidebar lets you specify a subset of important features (e.g., `GrLivArea`, `TotalBsmtSF`, `YearBuilt`, `Neighborhood`, `OverallQual`, `ExterQual`, `KitchenQual`, etc.).
- Any additional features the model expects but are not in the UI are automatically filled with `NaN` and imputed by the pipeline.
- When you click **"Predict Sale Price"**, the app:
	- Builds a single-row `DataFrame` with the correct training column names.
	- Calls `model.predict(...)` to obtain a log-price prediction.
	- Converts log-price back to the original scale with `np.expm1` and displays the dollar price.

If you retrain or re-tune the model in the notebook, you can overwrite `best_ames_model_fe.joblib` and the app will start using the updated pipeline automatically.

---
