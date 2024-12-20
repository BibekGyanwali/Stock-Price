import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import tensorflow as tf
from keras.models import load_model
model=load_model(r"C:\Users\dell\OneDrive\Desktop\Stock\Stock Price Prediction.keras")
import streamlit as st
st.title ("Stock Price Predictor")
stock=st.text_input("Enter Stock Symbol","TSLA")
start="2014-01-01"
end="2023-12-31"
Tesla=yf.download(stock,start,end)
st.subheader("Data")
st.write(Tesla)
Tesla_train=pd.DataFrame(Tesla.Close[0:int(len(Tesla)*0.80)])
Tesla_test=pd.DataFrame(Tesla.Close[int(len(Tesla)*0.80):len(Tesla)])
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler(feature_range=(0,1))
Last_100_days=Tesla_train.tail(100)
Tesla_test=pd.concat([Last_100_days,Tesla_test],ignore_index=True)
Tesla_test_scale=scaler.fit_transform(Tesla_test)
st.subheader("Price VS Moving average 50 days")
Mavg_50=Tesla.Close.rolling(50).mean()
fig1=plt.figure(figsize=(10,5))
plt.plot(Mavg_50,"red",label="Mavg 50 days")
plt.plot(Tesla.Close,"green",label="Closing Price")
plt.legend()
plt.show()
st.pyplot(fig1)
st.subheader("Price VS Moving average 50 days VS Moving average 100 days")
Mavg_100=Tesla.Close.rolling(100).mean()
fig2=plt.figure(figsize=(10,5))
plt.plot(Mavg_50,"red",label="Mavg 50 days")
plt.plot(Mavg_100,"blue",label="Mavg 100 days")
plt.plot(Tesla.Close,"green",label="Closing Price")
plt.legend()
plt.show()
st.pyplot(fig2)
st.subheader("Price VS Moving average 100 days VS Moving average 200 days")
Mavg_200=Tesla.Close.rolling(200).mean()
fig3=plt.figure(figsize=(10,5))
plt.plot(Mavg_100,"red",label="Mavg 100 days")
plt.plot(Mavg_200,"blue",label="Mavg 200 days")
plt.plot(Tesla.Close,"green",label="Closing Price")
plt.legend()
plt.show()
st.pyplot(fig3)
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
fig4=plt.figure(figsize=(10,5))
plt.plot(y_pred,'red',label='Predicted Price')
plt.plot(y,'green',label='Actual Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()
st.pyplot(fig4)
