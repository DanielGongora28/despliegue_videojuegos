#!/usr/bin/env python
# coding: utf-8

# # Despliegue
# 
# - Cargamos el modelo
# - Cargamos los datos futuros
# - Preparar los datos futuros
# - Aplicamos el modelo para la predicción

# In[3]:


#Cargamos librerías principales
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[4]:


#Cargamos el modelo
import pickle
filename = 'modelos_regresion.pkl'
model_Tree, model_Knn, model_NN,model_SVM, min_max_scaler, variables = pickle.load(open(filename, 'rb'))


# In[20]:


#Cargamos los datos futuros
# ? data = pd.read_csv("datos/videojuegos-datosFuturos.csv")
# ? data.head()


# In[ ]:


#Se crea interfaz gráfica con streamlit para captura de los datos

import streamlit as st

st.title('Predicción de inversión en una tienda de videojuegos')

Edad = st.slider('Edad', min_value=14, max_value=52, value=20, step=1)
videojuego = st.selectbox('Videojuego', ["'Mass Effect'","'Battlefield'", "'Fifa'","'KOA: Reckoning'","'Crysis'","'Sim City'","'Dead Space'","'F1'"])
Plataforma = st.selectbox('Plataforma', ["'Play Station'", "'Xbox'","PC","Otros"])
Sexo = st.selectbox('Sexo', ['Hombre', 'Mujer'])
Consumidor_habitual = st.selectbox('Consumidor_habitual', ['True', 'False'])


#Dataframe
datos = [[Edad, videojuego,Plataforma,Sexo,Consumidor_habitual]]
data = pd.DataFrame(datos, columns=['Edad', 'videojuego','Plataforma','Sexo','Consumidor_habitual']) #Dataframe con los mismos nombres de variables


# In[11]:


#Se realiza la preparación
data_preparada=data.copy()

#En despliegue drop_first= False
data_preparada = pd.get_dummies(data_preparada, columns=['videojuego', 'Plataforma','Sexo', 'Consumidor_habitual'], drop_first=False, dtype=int)
data_preparada.head()


# In[12]:


#Se adicionan las columnas faltantes
data_preparada=data_preparada.reindex(columns=variables,fill_value=0)
data_preparada.head()


# # **Predicciones**

# In[13]:


#Hacemos la predicción con el Tree
Y_Tree = model_Tree.predict(data_preparada)
print(Y_Tree)


# In[14]:


data['Prediccion Tree']=Y_Tree
data.head()


# In[15]:


#Se normaliza la edad para predecir con Knn, Red
#En los despliegues no se llama fit
data_preparada[['Edad']]= min_max_scaler.transform(data_preparada[['Edad']])
data_preparada.head()


# In[16]:


#Hacemos la predicción con Knn
Y_Knn = model_Knn.predict(data_preparada)
data['Prediccion Knn']=Y_Knn
data.head()


# In[17]:


#Hacemos la predicción con NN
Y_NN = model_NN.predict(data_preparada)
data['Prediccion NN']=Y_NN
data.head()


# In[18]:


#Hacemos la predicción con SVR
Y_SVM = model_SVM.predict(data_preparada)
data['Prediccion SVM']=Y_SVM
data.head()


# In[19]:


#Predicciones finales
data

