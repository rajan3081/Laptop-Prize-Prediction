import streamlit as st
import pickle
import numpy as np

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# load data and model metrics
df = pickle.load(open('df.pkl','rb'))
metrics = {}
models = {}
if os.path.exists('models/metrics.json'):
    import json
    metrics = json.load(open('models/metrics.json','r'))
    # load model pipelines
    for fname in os.listdir('models'):
        if fname.endswith('.pkl'):
            model_name = fname.replace('.pkl','')
            if model_name in metrics:
                try:
                    models[model_name] = pickle.load(open(os.path.join('models', fname),'rb'))
                except Exception:
                    pass

st.title("Laptop Price Predictor & Model Comparison")

# Sidebar navigation
st.sidebar.title('Navigation')
page = st.sidebar.radio('Go to', ['Home/EDA','Model Comparison','Prediction'])

if page == 'Home/EDA':
    st.header('Dataset Preview')
    st.dataframe(df.head(10))
    st.markdown('**Summary Statistics**')
    st.dataframe(df.describe())
    st.markdown('**Price Distribution**')
    fig, ax = plt.subplots()
    sns.histplot(df['Price'], kde=True, ax=ax)
    st.pyplot(fig)
    st.markdown('**Company Price Median**')
    fig2, ax2 = plt.subplots()
    sns.barplot(x='Company', y='Price', data=df, estimator=np.median, ax=ax2)
    plt.xticks(rotation=45)
    st.pyplot(fig2)
    st.markdown('**PPI vs Price**')
    fig3, ax3 = plt.subplots()
    sns.scatterplot(x='ppi', y='Price', data=df, ax=ax3)
    st.pyplot(fig3)
    st.stop()

if page == 'Model Comparison':
    st.header('Model Metrics')
    if metrics:
        metrics_df = pd.DataFrame(metrics).T

        metrics_df = metrics_df.round(2)
        metrics_df = metrics_df.sort_values(by='r2_score', ascending=False)
        st.dataframe(metrics_df)
        # charts
        st.subheader('Model R2 Comparison')
        fig4, ax4 = plt.subplots()
        metrics_df['r2_score'].plot(kind='bar', ax=ax4)
        st.pyplot(fig4)
        st.subheader('Model MAE Comparison')
        fig5, ax5 = plt.subplots()
        metrics_df['mae'].plot(kind='bar', color='orange', ax=ax5)
        st.pyplot(fig5)
        # Recommended best model by R2 (higher is better)
        recommended = metrics_df.sort_values(by='r2_score', ascending=False).iloc[0]
        best_name = metrics_df.sort_values(by='r2_score', ascending=False).index[0]
        st.markdown(f"**Recommended Model:** {best_name} — R2: {recommended['r2_score']}, MAE: {recommended['mae']}")
    else:
        st.warning('Metrics not found. Please run laptop.py to train models and save metrics in models/metrics.json')

    
    if metrics:
        recommended = pd.DataFrame(metrics).T.round(2).sort_values(by='r2_score', ascending=False).iloc[0]
        best_name = pd.DataFrame(metrics).T.sort_values(by='r2_score', ascending=False).index[0]
        st.markdown(f"**Recommended Model:** {best_name} — R2: {recommended['r2_score']}, MAE: {recommended['mae']}")

if page == 'Prediction':
    st.header('Make a Prediction')
    # inputs for prediction
    company = st.selectbox('Brand',df['Company'].unique())
    type = st.selectbox('Type',df['TypeName'].unique())
    ram = st.selectbox('RAM(in GB)',sorted(df['Ram'].unique().tolist()))
    weight = st.number_input('Weight of the Laptop',value=float(df['Weight'].mean()))
    touchscreen = st.selectbox('Touchscreen',['No','Yes'])
    ips = st.selectbox('IPS',['No','Yes'])
    screen_size = st.slider('Scrensize in inches', 10.0, 18.0, 13.0)
    resolution = st.selectbox('Screen Resolution',sorted(['1920x1080','1366x768','1600x900','3840x2160','3200x1800','2880x1800','2560x1600','2560x1440','2304x1440']))
    cpu = st.selectbox('CPU',df['Cpu brand'].unique())
    hdd = st.selectbox('HDD(in GB)',[0,128,256,512,1024,2048])
    ssd = st.selectbox('SSD(in GB)',[0,8,128,256,512,1024])
    gpu = st.selectbox('GPU',df['Gpu brand'].unique())
    os = st.selectbox('OS',df['os'].unique())

    # model selection for prediction
    model_choices = list(models.keys())
    best_model = None
    if metrics:
        best_model = max(metrics.items(), key=lambda x: (x[1]['r2_score'], -x[1]['mae']))[0]
    if model_choices:
        default_index = model_choices.index(best_model) if best_model in model_choices else 0
        selected_model = st.selectbox('Select Model for Prediction', model_choices, index=default_index)
    else:
        st.error('No trained models found. Please run laptop.py')
        st.stop()

    if st.button('Predict Price'):
        ppi = None
        if touchscreen == 'Yes':
            ts = 1
        else:
            ts = 0
        if ips == 'Yes':
            ips_f = 1
        else:
            ips_f = 0
        X_res = int(resolution.split('x')[0])
        Y_res = int(resolution.split('x')[1])
        ppi = ((X_res**2) + (Y_res**2))**0.5/screen_size
        # Build query as a pandas DataFrame to preserve dtypes and feature names
        query_dict = {
            'Company': [company],
            'TypeName': [type],
            'Ram': [int(ram)],
            'Weight': [float(weight)],
            'Touchscreen': [int(ts)],
            'Ips': [int(ips_f)],
            'ppi': [float(ppi)],
            'Cpu brand': [cpu],
            'HDD': [int(hdd)],
            'SSD': [int(ssd)],
            'Gpu brand': [gpu],
            'os': [os]
        }
        query = pd.DataFrame(query_dict)
        model_pipe = models.get(selected_model)
        if model_pipe is None:
            st.error('Selected model pipeline not loaded')
        else:
            pred_log = model_pipe.predict(query)[0]
            pred_price = int(np.exp(pred_log))
            st.success(f'The predicted price of this configuration is {pred_price}')
            if metrics:
                st.info(f'Best model overall: {best_model} (higher R2 is better)')
            # Show Predicted vs Actual sample and residuals
            st.subheader('Predicted vs Actual (sample)')
            try:
                X_sample = df.drop(columns=['Price']).sample(n=200, random_state=1)
            except Exception:
                X_sample = df.drop(columns=['Price'])
            preds_log = model_pipe.predict(X_sample)
            preds_price = np.exp(preds_log)
            actuals = df.loc[X_sample.index, 'Price']
            fig6, ax6 = plt.subplots()
            sns.scatterplot(x=actuals, y=preds_price, ax=ax6)
            ax6.set_xlabel('Actual Price')
            ax6.set_ylabel('Predicted Price')
            st.pyplot(fig6)
            st.subheader('Residuals (Actual - Predicted)')
            res = actuals - preds_price
            fig7, ax7 = plt.subplots()
            sns.histplot(res, kde=True, ax=ax7)
            st.pyplot(fig7)
            # Show Predicted vs Actual sample and residuals
            st.subheader('Predicted vs Actual (sample)')
            try:
                X_sample = df.drop(columns=['Price']).sample(n=200, random_state=1)
            except Exception:
                X_sample = df.drop(columns=['Price'])
            preds_log = model_pipe.predict(X_sample)
            preds_price = np.exp(preds_log)
            actuals = df.loc[X_sample.index, 'Price']
            fig6, ax6 = plt.subplots()
            sns.scatterplot(x=actuals, y=preds_price, ax=ax6)
            ax6.set_xlabel('Actual Price')
            ax6.set_ylabel('Predicted Price')
            st.pyplot(fig6)
            st.subheader('Residuals (Actual - Predicted)')
            res = actuals - preds_price
            fig7, ax7 = plt.subplots()
            sns.histplot(res, kde=True, ax=ax7)
            st.pyplot(fig7)
