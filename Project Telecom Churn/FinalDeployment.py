
import pandas as pd
import streamlit as st
from sklearn.linear_model import LogisticRegression
from PIL import Image
import matplotlib.pyplot as plt

st.header('Model Deployment')
st.subheader('Logistic & Random Forest Classification')

st.sidebar.title('Telecom customer churn')



voice_mail_messages = st.sidebar.text_input('voice mail messages number')
day_mins = st.sidebar.text_input('insert day minutes')
evening_mins = st.sidebar.text_input('insert evening minutes')
night_mins = st.sidebar.text_input('insert night minutes')
international_mins = st.sidebar.text_input('insert international minutes')
customer_service_calls = st.sidebar.text_input('insert customer service calls number')
international_plan = st.sidebar.selectbox('international_plan',('1','0'))
international_calls = st.sidebar.text_input('insert international calls number')
total_charge = st.sidebar.text_input('insert total charge value')



df = pd.read_csv('telecommunications_churn.csv')
df.drop(['voice_mail_plan','day_charge','evening_charge','night_charge', 'international_charge',
        'account_length','day_calls','night_calls','evening_calls'],axis=1, inplace=True)



# Upsampling the data
from sklearn.utils import resample
df_majority = df[df['churn']==0]
df_minority = df[df['churn']==1]
df_minority_upsample = resample(df_minority, replace=True, n_samples=2850,random_state=123)
df_upsample = pd.concat([df_minority_upsample,df_majority])



# Applying Logistic Regression
from sklearn.preprocessing import StandardScaler
p=df_upsample.iloc[:,:-1]
q=df_upsample.iloc[:,-1]
scale=StandardScaler()
P=scale.fit_transform(p)
clf=LogisticRegression()
clf.fit(P,q)



from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier




# Applying Random Forest Classifier
kfold=KFold(n_splits=10, random_state=123,shuffle=True)
model=RandomForestClassifier(n_estimators=100, max_features=3)
score=cross_val_score(model,P,q, cv=kfold)
model.fit(P,q)

list = df.columns.tolist()
choice = st.multiselect("Choose Any Feature",list)
data = df[choice]
st.line_chart(data)
#st.bar_chart(data)
fig, ax = plt.subplots()
ax.hist(data, bins=20)
st.pyplot(fig)

# Defining prediction

if st.button("Predict churn"):
    output = model.predict([[voice_mail_messages,day_mins,evening_mins,night_mins,international_mins,customer_service_calls,
                            international_plan,international_calls,total_charge]])
#    st.success("The prediction is {}".format(output[0])
    st.write('Customer will churn' if output[0] == 1 else 'Customer is loyal')
    if output[0] == 0:
        image = Image.open('happycust.jpeg')
        st.image(image, caption='Customer is Loyal')
    else:
        image = Image.open('sadcust.jpeg')
        st.image(image, caption='Customer Will Churn')
    
    
    
    