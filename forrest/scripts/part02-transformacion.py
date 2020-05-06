# %% [markdown]
# # Transformando dataset de Forest


# %%
import pandas as pd
import matplotlib.pyplot as plt


# %% [markdown]
# ## Cargando dataset limpio

# %%
data_file="../data/intermediate/forest-limpio.pkl"
data_origen = pd.read_pickle(data_file)
data.shape 


# %% [markdown]
# ### Wildernes_Area
# Corresponde a las áreas silvestres con menos perturbaciones realizadas por el hombre, la covertura es natural y no posee intervenciones forestales.

# %%
df_temp = data[["Wilderness_Area1", "Wilderness_Area2", "Wilderness_Area3", "Wilderness_Area4"]]
x = df_temp.stack() 
var = pd.Series(pd.Categorical(x[x!=0].index.get_level_values(1)))
df_temp = pd.concat([df_temp, var], axis=1)
df_temp.columns=["Wilderness_Area1", "Wilderness_Area2", "Wilderness_Area3", "Wilderness_Area4", "Wilderness_Area"]
df_temp.drop(["Wilderness_Area1", "Wilderness_Area2", "Wilderness_Area3", "Wilderness_Area4"], axis=1, inplace=True)
df_temp

# %% [markdown]
# La distribución de las áreas silvestres corresponde a:

# %%
df_temp["Wilderness_Area"].value_counts()

# %% 
%matplotlib inline 
df_temp.groupby("Wilderness_Area")["Wilderness_Area"].count().plot(kind="barh", legend="Reverse")
plt.xlabel("Nro. de ocurrencias")
plt.ylabel("Tipo de área silvestre")
plt.show()

# %% 
# transformando la columna "Wilderness_Area" a int.
df_temp["Wilderness_Area"] = df_temp["Wilderness_Area"].str.replace("Wilderness_Area","") 
df_temp["Wilderness_Area"].value_counts()

# %%
data = pd.concat([data, df_temp], axis=1)


# %% [markdown]
# ## Almacenando la transformacion del dataset
# El formato del archivo es .pkl, este tipo de archivo
# almacena la estructura de los objetos, esto permite
# mantener todos las categorias almacenadas internamente
# por lo tanto, no debere realizar filtrados o cosas por el estilo
# nuevamente.

# %%
data_file="../data/intermediate/forest-categorizado.pkl"
data.to_pickle(data_file)
print(data_file, " almacenado.")
data_file="../data/intermediate/forest-categorizado.csv"
data.to_csv(data_file, index=False, header=True)
print(data_file, " almacenado.")


# %% [markdown]
# # Finalizo la transformacion del dataset de Forest



