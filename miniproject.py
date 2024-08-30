import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


st.title("Web app using streamlit")

st.image("streamlit.png",width=500)

st.title("Case study on Diamond Dataset")

data=sns.load_dataset("diamonds")
st.write("shape of a dataset",data.shape)

menu=st.sidebar.radio("Menu",["HOME","Prediction price"])
if menu=="HOME":
    st.image("diamond.png",width=550)

    st.header("Tabular Data of a diamond")
    if st.checkbox("Tabular Data"):
        st.table(data.head(150))

    st.header("Statistical summary of a Dataframe")
    if st.checkbox("Statistics"):
       st.table(data.describe())

    st.title("Graphs")
    graph=st.selectbox("Different types of graphs",["Scatter plot","Bar Graph","Histogram"])

    if graph=="Scatter Plot":
        value=st.slider("Filter data using carat",0,6)
        data=data.loc[data["carat"]>=value]
        fig,ax=plt.subplots(figsize=(10,5))
        sns.scatterplot(data=data,x="carat",y="price",hue="cut")
        st.pyplot(fig)
        
    if graph=="Bar Graph":
        fig,ax=plt.subplots(figsize=(3.5,2))
        sns.barplot(x="cut",y=data.cut.index,data=data)
        st.pyplot(fig)
        
    if graph=="Histogram":
        fig,ax=plt.subplots(figsize=(5,3))
        sns.displot(data.price,kde=True)
        st.pyplot(fig)

    
if menu=="Prediction price":
    st.title("Prediction price of a diamond")

from sklearn.linear_model import LinearRegression 
lr=LinearRegression()
X=np.array(data["carat"]).reshape(-1,1)
y=np.array(data["price"]).reshape(-1,1)
lr.fit(X,y)

value=st.number_input("Carat",0.20,5.01,step=0.15)
value=np.array(value).reshape(1,-1)
prediction=lr.predict(value)[0]
if st.button("price Prediction($)"):
    st.write(f"{prediction}")