# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 17:53:31 2020

@author: Hasan
"""
import pandas as pd
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
import pickle
from sklearn.preprocessing import StandardScaler

df=pd.read_csv("C:\\Users\\Hasan\\Downloads\\datasets_14872_228180_Admission_Predict.csv")
df.drop(columns="Serial No.",inplace=True)
x=df.drop(columns="Chance of Admit ")
y=df[["Chance of Admit "]]
scaler=StandardScaler().fit(x)
x_scaled=scaler.transform(x)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x_scaled,y,test_size=0.2)
model=LinearRegression(normalize=True)
model.fit(x_train,y_train)
filename1 = 'gre-admission-rfc-model.pkl'
pickle.dump(model, open(filename1, 'wb'))
filename2 = 'scaler.pkl'
pickle.dump(scaler, open(filename2, 'wb'))
