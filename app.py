import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Load model
MODEL_PATH = "best_ames_model_fe.joblib"
if not os.path.exists(MODEL_PATH):
    st.error("Model file not found. Please place best_ames_model_fe.joblib in this directory.")
    st.stop()

model = joblib.load(MODEL_PATH)

st.title("Ames Housing Price Predictor")
st.markdown("Enter house characteristics and get a predicted sale price.")

# A minimal set of columns we will ask the user for.
# All other training columns will be filled with NaN and imputed by the pipeline.
# IMPORTANT: These names MUST match the training DataFrame's columns.

USER_NUMERIC_COLS = [
    "Gr Liv Area",
    "Total Bsmt SF",
    "Year Built",
    "Year Remod/Add",
    "Full Bath",
    "Half Bath",
    "Bsmt Full Bath",
    "Bsmt Half Bath",
    "Yr Sold",
]

USER_CATEGORICAL_COLS = [
    "Neighborhood",
    "House Style",
    "Overall Qual",   # numeric but ordinal
    "Exter Qual",
    "Kitchen Qual",
]

# A helper to build a full row with all expected columns
def build_input_row(user_inputs: dict) -> pd.DataFrame:
    """
    user_inputs: dict of columns we collect from the UI.
    We then create a single-row DataFrame with ALL columns the model saw during training,
    setting unspecified ones to NaN or simple defaults.
    """
    # Extract expected columns from the model's preprocessor
    preprocess = model.named_steps["preprocess"]
    all_feature_cols = []

    for name, transformer, cols in preprocess.transformers_:
        if name == "remainder":
            continue
        all_feature_cols.extend(cols)

    # Deduplicate if needed
    all_feature_cols = list(dict.fromkeys(all_feature_cols))

    # Start row with NaNs/defaults for all columns
    row = {col: np.nan for col in all_feature_cols}

    # Update with user-provided values (matching column names!)
    for k, v in user_inputs.items():
        if k in row:
            row[k] = v

    # Return as DataFrame
    return pd.DataFrame([row])


def main():
    st.sidebar.header("House Features")

    # Numeric inputs
    gr_liv_area = st.sidebar.number_input(
        "Above Ground Living Area (Gr Liv Area)",
        min_value=300,
        max_value=6000,
        value=1500,
    )
    total_bsmt_sf = st.sidebar.number_input(
        "Total Basement Area (Total Bsmt SF)",
        min_value=0,
        max_value=4000,
        value=800,
    )
    year_built = st.sidebar.number_input(
        "Year Built (Year Built)", min_value=1870, max_value=2025, value=1990
    )
    year_remod = st.sidebar.number_input(
        "Year Remodeled (Year Remod/Add)", min_value=1870, max_value=2025, value=1995
    )
    full_bath = st.sidebar.number_input(
        "Full Bathrooms (Full Bath)", min_value=0, max_value=5, value=2
    )
    half_bath = st.sidebar.number_input(
        "Half Bathrooms (Half Bath)", min_value=0, max_value=3, value=1
    )
    bsmt_full_bath = st.sidebar.number_input(
        "Bsmt Full Baths (Bsmt Full Bath)", min_value=0, max_value=3, value=0
    )
    bsmt_half_bath = st.sidebar.number_input(
        "Bsmt Half Baths (Bsmt Half Bath)", min_value=0, max_value=3, value=0
    )
    year_sold = st.sidebar.number_input(
        "Year Sold (Yr Sold)", min_value=2006, max_value=2025, value=2010
    )

    # Categorical / ordinal inputs
    neighborhood = st.sidebar.text_input("Neighborhood", "NAmes")
    house_style = st.sidebar.text_input("House Style", "1Story")
    overall_qual = st.sidebar.slider(
        "Overall Quality (Overall Qual)", min_value=1, max_value=10, value=5
    )
    exter_qual = st.sidebar.selectbox(
        "Exterior Quality (Exter Qual)", ["Po", "Fa", "TA", "Gd", "Ex"], index=2
    )
    kitchen_qual = st.sidebar.selectbox(
        "Kitchen Quality (Kitchen Qual)", ["Po", "Fa", "TA", "Gd", "Ex"], index=2
    )

    # Build dict with EXACT training column names
    user_inputs = {
        "Gr Liv Area": gr_liv_area,
        "Total Bsmt SF": total_bsmt_sf,
        "Year Built": year_built,
        "Year Remod/Add": year_remod,
        "Full Bath": full_bath,
        "Half Bath": half_bath,
        "Bsmt Full Bath": bsmt_full_bath,
        "Bsmt Half Bath": bsmt_half_bath,
        "Yr Sold": year_sold,
        "Neighborhood": neighborhood,
        "House Style": house_style,
        "Overall Qual": overall_qual,
        "Exter Qual": exter_qual,
        "Kitchen Qual": kitchen_qual,
    }

    input_df = build_input_row(user_inputs)

    if st.button("Predict Sale Price"):
        # Predict log price then convert back
        pred_log = model.predict(input_df)[0]
        pred_price = np.expm1(pred_log)

        st.subheader("Predicted Sale Price")
        st.write(f"${pred_price:,.0f}")

        st.markdown("---")
        st.markdown("**Debug Info (optional)**")
        st.write("Input features passed to model:")
        st.dataframe(input_df)


if __name__ == "__main__":
    main()