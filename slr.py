import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
df=pd.read_csv('Salary Data.csv')
df
print(df.describe())
print(df.tail())
print(df.shape)
print(df.head())
num_cols=df.select_dtypes(include='number').columns
for i in num_cols:
    Q1 = df[i].quantile(0.25)
    Q3 = df[i].quantile(0.75)
    IQR = Q3 - Q1

    lower=Q1-1.5*IQR
    upper=Q3+1.5*IQR

    count=((df[i] < lower) | (df[i] > upper)).sum()
    print(i,'outliers',count)

# find missing value
df.isnull().sum()
# handle missing value
for i in df.columns:
    if df[i].dtypes=='number':
        Q1 = df[i].quantile(0.25)
        Q3 = df[i].quantile(0.75)
        IQR = Q3 - Q1
        lower=Q1-1.5*IQR
        upper=Q3+1.5*IQR
        outlier=((df[i] < lower) | (df[i] > upper)).sum()
        if outliers>0:
            df[i]=df[i].fillna(df[i].median())
        else:
            df[i]=df[i].fillna(df[i].mean())
    else:
        df[i]=df[i].fillna(df[i].mode()[0])

# spliting of the data
from sklearn.model_selection import train_test_split
X=df[['Experience']]
y=df[['Salary']]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
# train the model on training data
from sklearn.linear_model import LinearRegression
reg=LinearRegression()
model=reg.fit(X_train,y_train)
model
import joblib
joblib.dump(model,"salary_training_model.pkl")
print("model saved successfully")
