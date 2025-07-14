import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

matplotlib.use('Agg')

st.title('Streamlit Advanced Application')

# I am adding this new line
def get_params(algo_name):
    params = dict()
    if algo_name == 'KNN':
        params['k'] = st.sidebar.slider('k', 1, 15)
    elif algo_name == 'SVM':
        params['C'] = st.sidebar.slider('C', 0.01, 10.0)
    return params


def get_classifier(algo_name, params):
    if algo_name == 'SVM':
        clf = SVC(C=params['C'])
    elif algo_name == 'KNN':
        clf = KNeighborsClassifier(n_neighbors=params['k'])
    elif algo_name == 'LR':
        clf = LogisticRegression()
    elif algo_name == 'Naive_bayes':
        clf = GaussianNB()
    else:
        st.error('choose an algorithm')

    return clf


def main():
    data = st.sidebar.file_uploader('Upload dataset', type=['csv'])
    if data is not None:
        st.success('Data upload complete')
        df = pd.read_csv(data)
        st.dataframe(df.head())
    activities = ['EDA', 'Visualization', 'Model']
    selection = st.sidebar.selectbox('Select your choice', activities)
    if selection == 'EDA':
        st.subheader('Exploratory Data Analysis')
        if st.checkbox('Display shape'):
            st.write(df.shape)
        if st.checkbox('Display columns'):
            st.write(df.columns)
        if st.checkbox('select multiple columns'):
            selected_cols = st.multiselect('Select columns:', df.columns)
            df1 = df[selected_cols]
            st.dataframe(df1.head())
        if st.checkbox('Display summary'):
            st.write(df.describe().T)
        if st.checkbox('Check for null values'):
            st.write(df.isnull().sum())
        if st.checkbox('Display correlation'):
            st.write(df.corr())

    elif selection == 'Visualization':
        st.subheader('Data Visualization')
        if data is not None:
            if st.checkbox('Display heatmap'):
                fig = plt.figure()
                sns.heatmap(df.corr(), vmax=1, square=True, annot=True, cmap='viridis')
                st.pyplot(fig)
            if st.checkbox('Display pairplot'):
                fig = plt.figure()
                sns.pairplot(data=df, hue='class')
                st.pyplot(fig)
    elif selection == 'Model':
        st.subheader('Machine Learning Modeling')
        if data is not None:
            x = df.iloc[:, 0:-1]
            y = df.iloc[:, -1]
            seed = st.sidebar.slider('seed', 1, 200)
            model_name = st.sidebar.selectbox('Select Algorithm', ('KNN', 'SVM', 'Naive_bayes', 'LR'))
            params = get_params(model_name)
            classifier = get_classifier(model_name, params)
            # Train test split of dataset
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)
            classifier.fit(x_train, y_train)
            y_pred = classifier.predict(x_test)
            st.write('Confusion Matrix')
            cm = metrics.confusion_matrix(y_test, y_pred)
            fig = plt.figure()
            sns.heatmap(cm, annot=True)
            st.pyplot(fig)
            st.write('Accuracy:', metrics.accuracy_score(y_test, y_pred))


if __name__ == main():
    main()
