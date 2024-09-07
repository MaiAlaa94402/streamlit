import streamlit as st
import pandas as pd
from pandas import Series, DataFrame
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
st.set_page_config(layout='wide')

df = pd.read_csv('customer_churn.csv')
churn = pd.get_dummies(df['Churn'],drop_first=True)

df = pd.concat([df,churn],axis=1)
df.rename(columns = {'Yes':'churn'}, inplace=True)

PhoneService_unique = df['PhoneService'].unique()
Multilines_unique = df['MultipleLines'].unique()
InternetService_unique = df['InternetService'].unique()
PaymentMethod_unique = df['PaymentMethod'].unique()
Contract_unique = df['Contract'].unique()

def DataDescription():
    st.title('Data Description')
    st.subheader('Sample Data')
    st.write(df.head())
    col1, col2= st.columns(2)
    
    with col1:
        st.subheader('Statistical summary')
        st.write(df.describe())
    with col2:
        st.subheader('Catigorical analysis')
        c1, c2 = st.columns(2)
        with c1:
            st.write("Internet services types:")
            st.write(InternetService_unique)

        with c2:
            st.write("Contarct types:")
            st.write(Contract_unique)
        co1, co2, co3 = st.columns([1,2,1])
        with co2:
            st.write("Payment methods:")
            st.write(PaymentMethod_unique)
            

        
    
def Visualization():
    st.title('Visualizations')
    st.header('Catigorical comparison')
    columns = ['gender','SeniorCitizen', 'Partner', 'InternetService', 'PhoneService', 'MultipleLines', 'PaymentMethod', 'PaperlessBilling', 'Contract']
    vis_choice = st.sidebar.selectbox('Catigorical comparison feature', columns)
    st.subheader(vis_choice)
    st.pyplot(SeabornCharts(vis_choice))
    
    st.header('Distribution comparison')
    tabs = st.tabs(['Box Plot', 'Distribution Plot'])
    columns_2 = ['tenure', 'MonthlyCharges']
    box_curve = st.sidebar.selectbox('Distribution comparison feature', columns_2)
    f1, f2 = (DistributionComparison(box_curve))
    with tabs[0]:
        st.subheader(box_curve)
        st.pyplot(f2)
    with tabs[1]:
        st.subheader(box_curve) 
        st.pyplot(f1)
    
def PredictionModle():
    st.title('Prediction Model')
    churn_dict = {
        0: 'Current customer',
        1: 'Former customer'
    }
    # Load the model
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
        
    Tenure = st.number_input('Tenure number', min_value = 0)
    MonthlyCharges = st.number_input('Monthly Charges', min_value = 0)
    PhoneService = st.selectbox('Do you have phone service?', ['Yes','No'])
    Multilines = st.selectbox('Do you have multi lines?', ['Yes','No'])
    InternetService = st.selectbox('What type of internet service you have?', ['DSL', 'Fiber Optic', 'No'])
    PaymentMethod = st.selectbox('What payment method you use?', ['Mailed Check', 'Electronic Check', 'Bank Transfer', 'Credit Card'])
    Contract = st.selectbox('What contract type you signed?', ['Month-to-month', 'One year', 'Two year'])
    
    button = st.button('Predict')
    
    if button:
        input_feature = FormatInput(Tenure, MonthlyCharges, PhoneService, Multilines, InternetService, PaymentMethod, Contract)

        prediction = model.predict(input_feature)

        st.write(f'Prediction result = {churn_dict[prediction[0]]}')
    
    
    
    
func_dict = {
    'Data Description' : DataDescription,
    'Data Analysis' : Visualization,
    'Predict' : PredictionModle
}

def SeabornCharts(col):
    former = df[df['Churn'] == 'Yes']
    current = df[df['Churn'] == 'No']
    
        
    fig, axes = plt.subplots(1, 3, figsize=(18,6))
    fig.suptitle(col.capitalize())
    
    if col == 'PaymentMethod':
        for ax in axes:
            ax.set_xticklabels(ax.get_xticklabels(), rotation=90)

    sns.countplot(ax=axes[0],x=former[col],palette="Pastel1", data=former).set(title='Former Customers')
    sns.countplot(ax=axes[1],x=current[col],palette="Pastel1", data=current).set(title='Current Customers')
    sns.pointplot(ax=axes[2],x=df[col],y=df["churn"],color='#7fcdbb',data=df)
    
    
    return fig

def DistributionComparison(col):
    fig = sns.FacetGrid(df,hue="Churn", palette='Set2',aspect=4)
    fig.map(sns.kdeplot, col, shade=True)

    oldest = df[col].max()

    fig.set(xlim=(0,oldest))

    fig.add_legend()
    
    fig2, ax = plt.subplots(figsize=(5, 3))
    sns.boxplot(x="Churn", y=col,palette='Set3',data=df)
    
    return fig, fig2

def FormatInput(Tenure, MonthlyCharges, PhoneService, Multilines, InternetService, PaymentMethod, Contract):
    phone = 0
    no_phone = 0
    multiline = 0
    no_multiline = 0
    DSL = 0
    Fiber = 0
    No_internet = 0
    bank_transfer = 0
    credit = 0
    electronic = 0
    mailed = 0
    m_t_m = 0
    one_y = 0
    twp_y = 0
    
    match PhoneService:
        case 'Yes':
            phone = 1
        case 'No':
            no_phone = 1
    
    match Multilines:
        case 'Yes':
            multiline = 1
        case'No':
            no_multiline = 1
    
    match InternetService:
        case 'DSL':
            DSL = 1
        case 'Fiber Optic':
            Fiber = 1
        case 'No':
            No_internet = 1
    match PaymentMethod:
        case 'Mailed Check':
            mailed = 1
        case 'Electronic Check':
            electronic = 1
        case 'Bank Transfer':
            bank_transfer = 1
        case 'Credit Card':
            credit = 1
    
    match Contract:
        case 'Month-to-month':
            m_t_m = 1
        case 'One year':
            one_y = 1
        case 'Two year':
            twp_y = 1
    result = np.array([
    [Tenure, MonthlyCharges, phone, no_multiline,
       no_phone, multiline, DSL, Fiber,
       No_internet, bank_transfer,
       credit, electronic, mailed,
       m_t_m, one_y, twp_y]
    ])
            
    return result
user_choice = st.sidebar.selectbox('Please select a page', func_dict.keys())
func_dict[user_choice]()
