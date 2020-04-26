# %% [markdown]
# # Explorando datos dataset cambio climático

# %%
import pandas as pd
import numpy as np
import  matplotlib.pyplot as plt 

# %%
data_file="../data/cambio_climatico.csv"
data = pd.read_csv(data_file)
data.shape 

# %%
data.describe(include="all")

# %%
data.info()

# %%
def buscar_nulos(dataframe):
    contador=0
    for columna in dataframe:
        nombre_col = dataframe[columna].name
        cantidad_col = len(dataframe[dataframe[columna].isnull()]) 
        porcentaje_col = (cantidad_col / (1.0*len(data))) * 100
        tipo_col = dataframe[columna].dtype
        if porcentaje_col != 0:
            contador+=1
            print(
                "Columna : {}".format(nombre_col), 
                "nulos {} ({}%)".format(cantidad_col, round(porcentaje_col)),
                "tipo {}".format(tipo_col), 
                sep="\t")
    if contador==0:
        print("No existen valores nulos en el dataframe.")
buscar_nulos(data)

# %%
data.dtypes

# %%
data["Year"].value_counts()

# %%
%matplotlib inline
k=int(np.ceil(1+np.log2(len(data))))
print("Cantidad de divisiones", k)
data.hist(bins=k, figsize=(20,15))
plt.show()

# %%
data["Temp"].hist(bins=k, figsize=(10,15))
plt.title("Histograma de temperatura")
plt.xlabel("Diferencia entre grados")
plt.ylabel("ocurrencias")
plt.show()

# %%
corr=data.corr()
corr

# %%
corr["Temp"].sort_values(ascending=True)

# %% 
%matplotlib inline
data.plot(kind="scatter",x="Temp", y="MEI")
plt.show()

# %%
data.plot(kind="scatter",x="Temp", y="Year")
plt.show()

# %% [markdown]
# ## Relación de las variables con la temperatura

# %%
data["Temp"].value_counts()
%matplotlib inline 
figure, axs = plt.subplots(2,3, sharey=True)
data.plot(kind="scatter", x="CO2", y="Temp", ax=axs[0][0])
data.plot(kind="scatter", x="CH4", y="Temp", ax=axs[0][1])
data.plot(kind="scatter", x="N2O",y="Temp", ax=axs[0][2])
data.plot(kind="scatter", x="CFC-11",y="Temp", ax=axs[1][0])
data.plot(kind="scatter", x="CFC-12",y="Temp", ax=axs[1][1])
data.plot(kind="scatter", x="TSI",y="Temp", ax=axs[1][2])
plt.show()

# %%
corr["Temp"]