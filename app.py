import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Load model
MODEL_PATH = 'best_ames_model_fe.joblib'
if not os.path.exists(MODEL_PATH):
    st.error('Model file not found. Please place best_ames_model_fe.joblib in this directory.')
    st.stop()

model = joblib.load(MODEL_PATH)

st.title('Ames Housing Price Predictor')
st.markdown('Enter house characteristics and get a predicted sale price.')

# For simplicity, we expose just a subset of key features.
# The pipeline will handle missing/unused columns internally.

def main():
    st.sidebar.header('House Features')

    # Numeric inputs (make sure these columns exist in training data)
    gr_liv_area = st.sidebar.number_input('Above Ground Living Area (GrLivArea)', min_value=300, max_value=6000, value=1500)
    total_bsmt_sf = st.sidebar.number_input('Total Basement Area (TotalBsmtSF)', min_value=0, max_value=4000, value=800)
    year_built = st.sidebar.number_input('Year Built', min_value=1870, max_value=2025, value=1990)
    year_remod = st.sidebar.number_input('Year Remodeled (YearRemodAdd)', min_value=1870, max_value=2025, value=1995)
    full_bath = st.sidebar.number_input('Full Bathrooms (FullBath)', min_value=0, max_value=5, value=2)
    half_bath = st.sidebar.number_input('Half Bathrooms (HalfBath)', min_value=0, max_value=3, value=1)
    bsmt_full_bath = st.sidebar.number_input('Bsmt Full Baths (BsmtFullBath)', min_value=0, max_value=3, value=0)
    bsmt_half_bath = st.sidebar.number_input('Bsmt Half Baths (BsmtHalfBath)', min_value=0, max_value=3, value=0)
    year_sold = st.sidebar.number_input('Year Sold (YrSold)', min_value=2006, max_value=2025, value=2010)

    # Simple categorical inputs
    neighborhood = st.sidebar.text_input('Neighborhood', 'NAmes')
    house_style = st.sidebar.text_input('HouseStyle', '1Story')
    overall_qual = st.sidebar.slider('Overall Quality (OverallQual)', 1, 10, 5)
    exter_qual = st.sidebar.selectbox('Exterior Quality (ExterQual)', ['Po','Fa','TA','Gd','Ex'], index=2)
    kitchen_qual = st.sidebar.selectbox('Kitchen Quality (KitchenQual)', ['Po','Fa','TA','Gd','Ex'], index=2)

    # Construct single-row DataFrame.
    # Any columns not specified here will be filled with NaN and imputed by the pipeline.
    input_dict = {
        'Gr Liv Area': gr_liv_area,
        'Total Bsmt SF': total_bsmt_sf,
        'Year Built': year_built,
        'Year Remod/Add': year_remod,
        'Full Bath': full_bath,
        'Half Bath': half_bath,
        'Bsmt Full Bath': bsmt_full_bath,
        'Bsmt Half Bath': bsmt_half_bath,
        'Yr Sold': year_sold,
        'Neighborhood': neighborhood,
        'House Style': house_style,
        'Overall Qual': overall_qual,
        'Exter Qual': exter_qual,
        'Kitchen Qual': kitchen_qual,
    }

    input_df = pd.DataFrame([input_dict])

    if st.button('Predict Sale Price'):
        # Predict log price then convert back
        pred_log = model.predict(input_df)[0]
        pred_price = np.expm1(pred_log)
        st.subheader('Predicted Sale Price')
        st.write(f'${pred_price:,.0f}')

        st.markdown('---')
        st.markdown('**Debug Info (optional)**')
        st.write('Input features:')
        st.dataframe(input_df)
