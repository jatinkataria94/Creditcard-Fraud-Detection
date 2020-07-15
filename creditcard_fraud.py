import timeit
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np
import warnings
warnings.filterwarnings("ignore")

import streamlit as st
st.title('Credit Card Fraud Detection!')

df=st.cache(pd.read_csv)('creditcard.csv')

#df = df.sample(frac=0.1, random_state = 48)

# Print shape and description of the data
if st.sidebar.checkbox('Show what the dataframe looks like'):
    st.write(df.head(100))
    st.write('Shape of the dataframe: ',df.shape)
    st.write('Data decription: \n',df.describe())


# Print valid and fraud transactions
fraud=df[df.Class==1]
valid=df[df.Class==0]
outlier_percentage=(df.Class.value_counts()[1]/df.Class.value_counts()[0])*100

if st.sidebar.checkbox('Show fraud and valid transaction details'):
    st.write('Fraudulent transactions are: %.3f%%'%outlier_percentage)
    st.write('Fraud Cases: ',len(fraud))
    st.write('Valid Cases: ',len(valid))

    
#Obtaining X (features) and y (labels)
X=df.drop(['Class'], axis=1)
y=df.Class

# Split the data into training and testing sets
from sklearn.model_selection import train_test_split
size = st.sidebar.slider('Test Set Size', min_value=0.2, max_value=0.4)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = size, random_state = 42)

#Print shape of train and test sets
if st.sidebar.checkbox('Show the shape of training and test set features and labels'):
    st.write('X_train: ',X_train.shape)
    st.write('y_train: ',y_train.shape)
    st.write('X_test: ',X_test.shape)
    st.write('y_test: ',y_test.shape)
    

#Import classification models and metrics
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier
from sklearn.model_selection import cross_val_score


logreg=LogisticRegression()
svm=SVC()
knn=KNeighborsClassifier()
etree=ExtraTreesClassifier(random_state=42)
rforest=RandomForestClassifier(random_state=42)


features=X_train.columns.tolist()


#Feature selection through feature importance
@st.cache
def feature_sort(model,X_train,y_train):
    #feature selection
    mod=model
    # fit the model
    mod.fit(X_train, y_train)
    # get importance
    imp = mod.feature_importances_
    return imp

#Classifiers for feature importance
clf=['Extra Trees','Random Forest']
mod_feature = st.sidebar.selectbox('Which model for feature importance?', clf)

start_time = timeit.default_timer()
if mod_feature=='Extra Trees':
    model=etree
    importance=feature_sort(model,X_train,y_train)
elif mod_feature=='Random Forest':
    model=rforest
    importance=feature_sort(model,X_train,y_train)
elapsed = timeit.default_timer() - start_time
st.write('Execution Time for feature selection: %.2f minutes'%(elapsed/60))    

#Plot of feature importance
if st.sidebar.checkbox('Show plot of feature importance'):
    plt.bar([x for x in range(len(importance))], importance)
    plt.title('Feature Importance')
    plt.xlabel('Feature (Variable Number)')
    plt.ylabel('Importance')
    st.pyplot()

feature_imp=list(zip(features,importance))
feature_sort=sorted(feature_imp, key = lambda x: x[1])

n_top_features = st.sidebar.slider('Number of top features', min_value=5, max_value=20)

top_features=list(list(zip(*feature_sort[-n_top_features:]))[0])

if st.sidebar.checkbox('Show selected top features'):
    st.write('Top %d features in order of importance are: %s'%(n_top_features,top_features[::-1]))

X_train_sfs=X_train[top_features]
X_test_sfs=X_test[top_features]

X_train_sfs_scaled=X_train_sfs
X_test_sfs_scaled=X_test_sfs



#Import performance metrics, imbalanced rectifiers
from sklearn.metrics import  confusion_matrix,classification_report,matthews_corrcoef
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss
np.random.seed(42) #for reproducibility since SMOTE and Near Miss use randomizations

smt = SMOTE()
nr = NearMiss()

def compute_performance(model, X_train, y_train,X_test,y_test):
    start_time = timeit.default_timer()
    scores = cross_val_score(model, X_train, y_train, cv=3, scoring='accuracy').mean()
    'Accuracy: ',scores
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    cm=confusion_matrix(y_test,y_pred)
    'Confusion Matrix: ',cm  
    cr=classification_report(y_test, y_pred)
    'Classification Report: ',cr
    mcc= matthews_corrcoef(y_test, y_pred)
    'Matthews Correlation Coefficient: ',mcc
    elapsed = timeit.default_timer() - start_time
    'Execution Time for performance computation: %.2f minutes'%(elapsed/60)
    

#Run different classification models with rectifiers
if st.sidebar.checkbox('Run a credit card fraud detection model'):
    
    alg=['Extra Trees','Random Forest','k Nearest Neighbor','Support Vector Machine','Logistic Regression']
    classifier = st.sidebar.selectbox('Which algorithm?', alg)
    rectifier=['SMOTE','Near Miss','No Rectifier']
    imb_rect = st.sidebar.selectbox('Which imbalanced class rectifier?', rectifier) 
    
    if classifier=='Logistic Regression':
        model=logreg
        if imb_rect=='No Rectifier':
            compute_performance(model, X_train_sfs_scaled, y_train,X_test_sfs_scaled,y_test)
        elif imb_rect=='SMOTE':
                rect=smt
                st.write('Shape of imbalanced y_train: ',np.bincount(y_train))
                X_train_bal, y_train_bal = rect.fit_sample(X_train_sfs_scaled, y_train)
                st.write('Shape of balanced y_train: ',np.bincount(y_train_bal))
                compute_performance(model, X_train_bal, y_train_bal,X_test_sfs_scaled,y_test)
        elif imb_rect=='Near Miss':
            rect=nr
            st.write('Shape of imbalanced y_train: ',np.bincount(y_train))
            X_train_bal, y_train_bal = rect.fit_sample(X_train_sfs_scaled, y_train)
            st.write('Shape of balanced y_train: ',np.bincount(y_train_bal))
            compute_performance(model, X_train_bal, y_train_bal,X_test_sfs_scaled,y_test)
    
        
    elif classifier == 'k Nearest Neighbor':
        model=knn
        if imb_rect=='No Rectifier':
            compute_performance(model, X_train_sfs_scaled, y_train,X_test_sfs_scaled,y_test)
        elif imb_rect=='SMOTE':
                rect=smt
                st.write('Shape of imbalanced y_train: ',np.bincount(y_train))
                X_train_bal, y_train_bal = rect.fit_sample(X_train_sfs_scaled, y_train)
                st.write('Shape of balanced y_train: ',np.bincount(y_train_bal))
                compute_performance(model, X_train_bal, y_train_bal,X_test_sfs_scaled,y_test)
        elif imb_rect=='Near Miss':
            rect=nr
            st.write('Shape of imbalanced y_train: ',np.bincount(y_train))
            X_train_bal, y_train_bal = rect.fit_sample(X_train_sfs_scaled, y_train)
            st.write('Shape of balanced y_train: ',np.bincount(y_train_bal))
            compute_performance(model, X_train_bal, y_train_bal,X_test_sfs_scaled,y_test)    
    
    elif classifier == 'Support Vector Machine':
        model=svm
        if imb_rect=='No Rectifier':
            compute_performance(model, X_train_sfs_scaled, y_train,X_test_sfs_scaled,y_test)
        elif imb_rect=='SMOTE':
                rect=smt
                st.write('Shape of imbalanced y_train: ',np.bincount(y_train))
                X_train_bal, y_train_bal = rect.fit_sample(X_train_sfs_scaled, y_train)
                st.write('Shape of balanced y_train: ',np.bincount(y_train_bal))
                compute_performance(model, X_train_bal, y_train_bal,X_test_sfs_scaled,y_test)
        elif imb_rect=='Near Miss':
            rect=nr
            st.write('Shape of imbalanced y_train: ',np.bincount(y_train))
            X_train_bal, y_train_bal = rect.fit_sample(X_train_sfs_scaled, y_train)
            st.write('Shape of balanced y_train: ',np.bincount(y_train_bal))
            compute_performance(model, X_train_bal, y_train_bal,X_test_sfs_scaled,y_test)    
        
    elif classifier == 'Random Forest':
        model=rforest
        if imb_rect=='No Rectifier':
            compute_performance(model, X_train_sfs_scaled, y_train,X_test_sfs_scaled,y_test)
        elif imb_rect=='SMOTE':
                rect=smt
                st.write('Shape of imbalanced y_train: ',np.bincount(y_train))
                X_train_bal, y_train_bal = rect.fit_sample(X_train_sfs_scaled, y_train)
                st.write('Shape of balanced y_train: ',np.bincount(y_train_bal))
                compute_performance(model, X_train_bal, y_train_bal,X_test_sfs_scaled,y_test)
        elif imb_rect=='Near Miss':
            rect=nr
            st.write('Shape of imbalanced y_train: ',np.bincount(y_train))
            X_train_bal, y_train_bal = rect.fit_sample(X_train_sfs_scaled, y_train)
            st.write('Shape of balanced y_train: ',np.bincount(y_train_bal))
            compute_performance(model, X_train_bal, y_train_bal,X_test_sfs_scaled,y_test)  
            
    elif classifier == 'Extra Trees':
        model=etree
        if imb_rect=='No Rectifier':
            compute_performance(model, X_train_sfs_scaled, y_train,X_test_sfs_scaled,y_test)
        elif imb_rect=='SMOTE':
                rect=smt
                st.write('Shape of imbalanced y_train: ',np.bincount(y_train))
                X_train_bal, y_train_bal = rect.fit_sample(X_train_sfs_scaled, y_train)
                st.write('Shape of balanced y_train: ',np.bincount(y_train_bal))
                compute_performance(model, X_train_bal, y_train_bal,X_test_sfs_scaled,y_test)
        elif imb_rect=='Near Miss':
            rect=nr
            st.write('Shape of imbalanced y_train: ',np.bincount(y_train))
            X_train_bal, y_train_bal = rect.fit_sample(X_train_sfs_scaled, y_train)
            st.write('Shape of balanced y_train: ',np.bincount(y_train_bal))
            compute_performance(model, X_train_bal, y_train_bal,X_test_sfs_scaled,y_test)
            
    
        



