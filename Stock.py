import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import load_model
model=load_model(r"C:\Users\dell\OneDrive\Desktop\Stock\Stock Price Prediction.keras")
Tesla=pd.read_csv(r"C:\Users\dell\Downloads\Tesla.csv")
import streamlit as st
st.title ("Stock Price Prediction")
st.header ("Tesla Stock Price")
Tesla_train=pd.DataFrame(Tesla.Close[0:int(len(Tesla)*0.80)])
Tesla_test=pd.DataFrame(Tesla.Close[int(len(Tesla)*0.80):len(Tesla)])
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler(feature_range=(0,1))
Last_100_days=Tesla_train.tail(100)
Tesla_test=pd.concat([Last_100_days,Tesla_test],ignore_index=True)
Tesla_test_scale=scaler.fit_transform(Tesla_test)
x=[]
y=[]
for i in range(100,Tesla_test_scale.shape[0]):
    x.append(Tesla_test_scale[i-100:i])
    y.append(Tesla_test_scale[i,0])
x,y=np.array(x),np.array(y)
y_pred=model.predict(x)
scale=1/scaler.scale_
y_pred=y_pred*scale
y=y*scale
st.header("Actual Value VS Predicted Value")
figure=plt.figure(figsize=(10,5))
plt.plot(y_pred,'red',label='Predicted Price')
plt.plot(y,'green',label='Actual Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()
st.pyplot(figure)
