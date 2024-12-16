import os
import streamlit as st
import joblib 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    fbeta_score
)


model_path = os.path.join(os.path.dirname(__file__), 'lgbmc_model.pkl')

X_train = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', 'X_train_imputed.csv'))
y_train = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', 'y_train.csv'))
X_test = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', 'X_test_imputed.csv'))
y_test = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', 'y_test.csv'))

lgbmc_model = joblib.load(model_path)

def preprocess_input(Tenure, WarehouseToHome, NumberOfDeviceRegistered, DaySinceLastOrder, CashbackAmount, NumberOfAddress, 
                     MaritalStatus, Complain, PreferedOrderCat, SatisfactionScore):
    
    input_data = pd.DataFrame({
        'Tenure': [Tenure],
        'WarehouseToHome': [WarehouseToHome],
        'NumberOfDeviceRegistered': [NumberOfDeviceRegistered],
        'DaySinceLastOrder': [DaySinceLastOrder],
        'CashbackAmount': [CashbackAmount],
        'NumberOfAddress': [NumberOfAddress],
        'MaritalStatus': [MaritalStatus],
        'Complain': [Complain],
        'PreferedOrderCat': [PreferedOrderCat],
        'SatisfactionScore': [SatisfactionScore]
    })
    
    return input_data

st.title('E-commerce Customer Churn Prediction')
st.sidebar.header('Enter Customer Information')

Tenure = st.sidebar.number_input('Tenure', min_value=0, max_value=61, value=0, step=1)
WarehouseToHome = st.sidebar.number_input('WarehouseToHome', 5, 127, 5, 1)
NumberOfDeviceRegistered = st.sidebar.number_input('NumberOfDeviceRegistered', 1, 6, 1, 1)
DaySinceLastOrder = st.sidebar.number_input('DaySinceLastOrder', 0, 46, 0, 1)
CashbackAmount = st.sidebar.number_input('CashbackAmount', 0.0, 325.0, 0.0, 0.1)
NumberOfAddress = st.sidebar.number_input('NumberOfAddress', 1, 22, 1, 1)

MaritalStatus = st.sidebar.selectbox('MaritalStatus', ('Single', 'Married', 'Divorced'))
Complain = st.sidebar.selectbox('Complain', (0, 1))
PreferedOrderCat = st.sidebar.selectbox('PreferedOrderCat', ('Laptop & Accessory', 'Mobile', 'Fashion', 'Grocery', 'Others'))
SatisfactionScore = st.sidebar.selectbox('SatisfactionScore', (1, 2, 3, 4, 5))

input_data = preprocess_input(Tenure, WarehouseToHome, NumberOfDeviceRegistered, DaySinceLastOrder, CashbackAmount, 
                              NumberOfAddress, MaritalStatus, Complain, PreferedOrderCat, SatisfactionScore)

st.markdown('<br><br>', unsafe_allow_html=True)

st.markdown('---')
st.header('Single Prediction')
st.markdown('---')

prediction = lgbmc_model.predict(input_data)
pred_proba = lgbmc_model.predict_proba(input_data)

if prediction == 0:
    st.subheader(f'Good news! This customer is likely to continue using your product with a probability of {pred_proba[0][0] * 100:.2f}% 🎉.')
else:
    st.subheader(f'Unfortunately, this customer may churn with a probability of {pred_proba[0][1] * 100:.2f}%. Act fast to retain them 🏃!')

st.markdown('<br><br>', unsafe_allow_html=True)

st.markdown('---')
st.header('Batch Prediction')
st.markdown('---')

uploaded_file = st.file_uploader('Upload a CSV file for batch prediction', type=['csv'])

if uploaded_file:
    batch_data = pd.read_csv(uploaded_file)
    st.write('Uploaded Data:')
    st.write(batch_data)

    batch_predictions = lgbmc_model.predict(batch_data)
    batch_data['Prediction'] = batch_predictions

    st.subheader('Prediction Results:')
    st.write(batch_data)

    churn_count = (batch_predictions == 1).sum()
    non_churn_count = (batch_predictions == 0).sum()
    total_users = len(batch_predictions)
    churn_percentage = (churn_count / total_users) * 100
    non_churn_percentage = (non_churn_count / total_users) * 100

    fig, ax = plt.subplots()
    ax.pie([non_churn_count, churn_count], labels=['Not Churn', 'Churn'], autopct='%.2f%%', startangle=90, colors=['#5cb85c', '#d9534f'])
    ax.axis('equal')
    st.pyplot(fig)

    if churn_count > non_churn_count:
        st.subheader(f'Alert! {churn_count} users (or {churn_percentage:.2f}% of your data) are likely to churn 😱. Take immediate action to retain them!')
    else:
        st.subheader(f'Great news! Approximately {non_churn_count} users (or {non_churn_percentage:.2f}% of your data) are not likely to churn 🎉. Keep up the excellent work, and act quickly to retain those at risk of churning!')

st.markdown('<br><br>', unsafe_allow_html=True)

st.markdown('---')
st.header('Model Performance')
st.markdown('---')

y_train_pred = lgbmc_model.predict(X_train)
y_test_pred = lgbmc_model.predict(X_test)

def calculate_metrics(y_true, y_pred, beta=2):
    y_true = np.array(y_true, dtype=int)
    y_pred = np.array(y_pred, dtype=int)

    metrics = {
        'Class 0': {
            'Precision': precision_score(y_true, y_pred, pos_label=0),
            'Recall': recall_score(y_true, y_pred, pos_label=0),
            'F1-Score': f1_score(y_true, y_pred, pos_label=0),
            'F2-Score': fbeta_score(y_true, y_pred, beta=beta, pos_label=0)
        },
        'Class 1': {
            'Precision': precision_score(y_true, y_pred, pos_label=1),
            'Recall': recall_score(y_true, y_pred, pos_label=1),
            'F1-Score': f1_score(y_true, y_pred, pos_label=1),
            'F2-Score': fbeta_score(y_true, y_pred, beta=beta, pos_label=1)
        },
        'Macro Average': {
            'Precision': precision_score(y_true, y_pred, average='macro'),
            'Recall': recall_score(y_true, y_pred, average='macro'),
            'F1-Score': f1_score(y_true, y_pred, average='macro'),
            'F2-Score': fbeta_score(y_true, y_pred, beta=beta, average='macro')
        },
        'Weighted Average': {
            'Precision': precision_score(y_true, y_pred, average='weighted'),
            'Recall': recall_score(y_true, y_pred, average='weighted'),
            'F1-Score': f1_score(y_true, y_pred, average='weighted'),
            'F2-Score': fbeta_score(y_true, y_pred, beta=beta, average='weighted')
        }
    }
    return metrics

def display_metrics(metrics, title):
    st.subheader(title)
    metrics_df = pd.DataFrame(metrics).T
    st.dataframe(metrics_df)

def plot_confusion_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Class 0', 'Class 1'], yticklabels=['Class 0', 'Class 1'])
    ax.set_title(title)
    ax.set_xlabel('Predicted Labels')
    ax.set_ylabel('True Labels')
    st.pyplot(fig)

train_metrics = calculate_metrics(y_train, y_train_pred)
test_metrics = calculate_metrics(y_test, y_test_pred)

display_metrics(train_metrics, 'Classification Report (Train)')
plot_confusion_matrix(y_train, y_train_pred, 'Confusion Matrix (Train)')

display_metrics(test_metrics, 'Classification Report (Test)')
plot_confusion_matrix(y_test, y_test_pred, 'Confusion Matrix (Test)')