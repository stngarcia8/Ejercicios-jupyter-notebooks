# %% [markdown]
# # Evaluación 02 -  Scatters
# Alumno: Daniel García Loyola<br>
# Profesor: Israel Naranjo Retamal<br>
# Ramo: Minería de datos (MDY7101)<br>
# Sección: 002D<br>
# Entrega: 12/06/2020<br>
# Pregunta: 02


# %%

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings


# %%
# Ajustes al notebook
warnings.filterwarnings('ignore')
#matplot inline

# %% [markdown]
# # Cargando datos de skatters

# %%
data_file = "../data/original/Skate.csv"
data = pd.read_csv(data_file, engine='python')
data.shape

# %% [markdown]
# Ajustando nombres de columnas.

# %%
data.columns = [
    'ID', 'NombreAtleta', 'Edad', 'Nacionalidad', 'Puntos',
    'Ev11', 'Ev12', 'Ev13', 'Ev14', 'Ev15', 'Ev16',
    'Ev21', 'Ev22', 'Ev23', 'Ev24', 'Ev25', 'Ev26']
data.columns

# %% [markdown]
# # Explorando datos


# %%
data.head()

# %%
data.describe(include='all')

# %%
data.info()

# %%
data.dtypes


# %% [markdown]
# # Analisis y procesado de datos.
# Se realizarán ajustes al dataframe para que el comportamiento frente a los modelos predictivos sea óptimo.

# %% [markdown]
# ## Procesando elementos nulos


# %%
print("Cantidad de elementos nulos por columna:")
data.isnull().sum()


# %% [markdown]
# Los elementos nulos del dataframe se encuentran concentrados en las columnas
# de eventos (EvX), esto puede indicar dos cosas,
# primero si una columna posee todos sus elementos con valor cero significa que no se ha realizado y en el caso
# que los elementos sean algunos dentro de la columnas significará que
# el participante no compitió en dicho evento ya que no posee un puesto en dicho evento.

# %%
data.fillna(0, inplace=True)
data.head()


# %%
print("Cantidad de elementos nulos por columna:")
data.isnull().sum()



# %% [markdown]
# # Procesando el dataframe

# %% [markdown]
# ## Calculando total de participaciones


# %%
columnas = [
    'Ev11', 'Ev12', 'Ev13', 'Ev14', 'Ev15', 'Ev16',
    'Ev21', 'Ev22', 'Ev23', 'Ev24', 'Ev25', 'Ev26']
for col in columnas:
    col_tmp = 'p' + str(col).strip()
    data[col_tmp] = 0
    data.loc[data[col]>0,  col_tmp] =1


# %%
# Calcular el total de participaciones
columnas = [
    'pEv11', 'pEv12', 'pEv13', 'pEv14', 'pEv15', 'pEv16',
    'pEv21', 'pEv22', 'pEv23', 'pEv24', 'pEv25', 'pEv26']
data['Participaciones'] = data[columnas].apply(lambda x: (x.pEv11+x.pEv12+x.pEv13+x.pEv14+x.pEv15+x.pEv16+x.pEv21+x.pEv22+x.pEv23+x.pEv24+x.pEv25+x.pEv26), axis=1)
data.head(5)



# %% [markdown]
# # Calculando participaciones separados por top

# %%
def contar_participaciones(columnas, valores):
    for col in columnas:
        col_tmp = 'p' + str(col).strip()
        data[col_tmp] = 0
        data.loc[data[col].isin(valores),  col_tmp] =1
    return


# %%
def calcular_participaciones(columnas, columna):
    data[columna] = data[columnas].apply(lambda x: (x.pEv11+x.pEv12+x.pEv13+x.pEv14+x.pEv15+x.pEv16+x.pEv21+x.pEv22+x.pEv23+x.pEv24+x.pEv25+x.pEv26), axis=1)
    return


# %% [markdown]
# ## Calculando participaciones top 10


# %%
columnas = [
    'Ev11', 'Ev12', 'Ev13', 'Ev14', 'Ev15', 'Ev16',
    'Ev21', 'Ev22', 'Ev23', 'Ev24', 'Ev25', 'Ev26']
p_columnas = [
    'pEv11', 'pEv12', 'pEv13', 'pEv14', 'pEv15', 'pEv16',
    'pEv21', 'pEv22', 'pEv23', 'pEv24', 'pEv25', 'pEv26']
valores = [ i for i in range(1, 11)]
contar_participaciones(columnas, valores)
calcular_participaciones(p_columnas, 'WinTop10')


# %% [markdown]
# ## Calculando participaciones top 20


# %%
valores = [ i for i in range(11, 21)]
contar_participaciones(columnas, valores)
calcular_participaciones(p_columnas, 'WinTop20')


# %% [markdown]
# ## Calculando participaciones top 30


# %%
valores = [ i for i in range(21, 31)]
contar_participaciones(columnas, valores)
calcular_participaciones(p_columnas, 'WinTop30')


# %% [markdown]
# ## Calculando participaciones top 30


# %%
valores = [ i for i in range(31, 41)]
contar_participaciones(columnas, valores)
calcular_participaciones(p_columnas, 'WinTop40')


# %% [markdown]
# ## Calculando participaciones top 40


# %%
valores = [ i for i in range(41, 51)]
contar_participaciones(columnas, valores)
calcular_participaciones(p_columnas, 'WinTop50')


# %% [markdown]
# ## Calculando participaciones top mayores a 50 hasta 100


# %%
valores = [ i for i in range(51, 101)]
contar_participaciones(columnas, valores)
calcular_participaciones(p_columnas, 'WinTop100')



# %% [markdown]
# ## Eliminando columnas inecesarias

# %%
data.drop(columnas, axis=1, inplace=True)
data.drop(p_columnas, axis=1, inplace=True)
data.head()


# %% [markdown]
# # Visualizando el estado de las columnas

# %%
data_corr = data.corr()
sns.heatmap(data_corr)


# %%
data_corr


# %% [markdown]
# # Visualizando relación de puntajes y participaciones


# %%
plt.title('Distribución de puntos vs participaciones')
sns.lmplot(x='Puntos', y='Participaciones', hue='WinTop10', data=data)
plt.show()


# %% [markdown]
# # Obteniendo los datos de la atleta a buscar

# %%
nombre = data.loc[:, 'NombreAtleta'].str.contains('Maria Jose Rojas')
maria = data.loc[nombre]
maria


# %% [markdown]
# hemos preparado el dataframe para usarlo en el modelo
# K-Means, ahora se debe seguir los pasos pertinentes para
# que los datos sean procesados correctamente por el modelo.


# %% [markdown]
# # Preparando el modelo


# %%
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn import preprocessing


# %% [markdown]
# obteniendo los datos para el modelo


# %%
x = data.iloc[:, [2, 4, 5]].values
print(x)


# %% [markdown]
# Buscando el número óptimos de grupos

# %%
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)
#Plotting the results onto a line graph, allowing us to observe 'The elbow'
plt.plot(range(1, 11), wcss)
plt.title('Buscando número óptimo de clusters')
plt.xlabel('Número de clusters')
plt.ylabel('WCSS')
plt.show()


# %%
kmeans = KMeans(n_clusters = 10, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
y_kmeans = kmeans.fit_predict(x)


# %%
plt.scatter(x[y_kmeans == 0, 0], x[y_kmeans == 0, 1], s = 100, c = 'm', label = 'Edad')
plt.scatter(x[y_kmeans == 1, 0], x[y_kmeans == 1, 1], s = 100, c = 'c', label = 'puntos')
plt.scatter(x[y_kmeans == 2, 0], x[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Participaciones')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:,1], s = 100, c = 'k', label = 'Centroides')
plt.legend()

# %%
# dejando como caracteristicas la edad y las participaciones
X = data.iloc[:, [0, 2, 4, 5]]
print(X.sample(5))


# %%
# Seleccionando el puntaje como label
y = data.iloc[:, [1]]
print(y.sample(5))


# %% [markdown]
# Preparando el modelo finalmente


# %%
scaler = preprocessing.StandardScaler()
scaler.fit(X)
X_scaled_array = scaler.transform(X)
X_scaled = pd.DataFrame(X_scaled_array, columns = X.columns)
X_scaled.sample(5)


# %% [markdown]
# # Proyectando la participación


# %%
m = maria[['Edad', 'Puntos', 'Participaciones']]
y_pred = kmeans.predict(m)
nombre = maria['NombreAtleta'].values
probabilidad = 'no tiene' if y_pred[0] == 0 else 'aún posee'
sino = 'no' if y_pred[0] == 0 else ''
print('{0} {1} opciones para asistir a los juegos olímpicos de Tokio 2020, sus caracteristicas {2} lo permiten'.format(nombre, probabilidad, sino))
print('Caracteristicas actuales:')
maria


# %% [markdown]
# # Conclusión
# En este ejercicio se ha utilizado el modelo no supervisado de 
# KMeans, el dataframe ha sido preparado para este tipo 
# de modelo, generando variables que permitan a este 
# poder predecir de mejor manera según los datos de una 
# competidora en especial (**Maria Jose Rojas**) 
# la cual será evaluada por el modelo según las 
# características que posea.


# %% [markdown]
# # Configuración utilizada para esta evaluación.
# CPython 3.7.6<br>
# IPython 7.13.0<br>
# compiler   : MSC v.1916 64 bit (AMD64)<br>
# system     : Windows<br>
# release    : 10<br>
# machine    : AMD64<br>
# processor  : Intel64 Family 6 Model 158 Stepping 10, GenuineIntel<br>
# CPU cores  : 12<br>
# interpreter: 64bit<br>
