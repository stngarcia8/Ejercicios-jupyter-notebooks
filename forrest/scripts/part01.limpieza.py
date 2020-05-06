# %% [markdown]
# # Limpieza de dataset de forrest

# %% [markdown]
# ## Objetivo
# Limpiar el dataset de forrest, permitiendo posteriormente un analisis de datos con la información optimizada.

# %% 
import pandas as pd

# %%
data_file="../data/original/forest.csv"
data = pd.read_csv(data_file)
data.shape 



# %% [markdown]
# ## Explorando datos de forest
data.describe()

# %% [markdown]
# ## Visualizando info del dataframe
data.info()

# %% [markdown]
# ## Verificando tipos de datos
data.dtypes



# %% [markdown]
# ## Visualizando elementos nulos

# %%
data.isnull().sum()



# %% [markdown]
# ## Definiendo variables categoricas
# Las variables covertura del bosque (cover_type), tipos de suelo (Soil_Type) y areas silvestres serán catalogadas como categoricas, ya que sus valores solo corresponden a 1 presente y 0 ausente.

# %%
def convertir_categoricas(df, columnas):
    for columna in columnas:
        df[columna] = df[columna].astype("category")
        print(columna, df[columna].dtype, sep="\t")


# %%
lista=["Wilderness_Area1", "Wilderness_Area2", "Wilderness_Area3", "Wilderness_Area4"]
convertir_categoricas(data, lista)





# %%
lista=["Cover_Type"]
convertir_categoricas(data, lista)


# %%
lista=["Soil_Type1", "Soil_Type2", "Soil_Type3", "Soil_Type4", "Soil_Type5", "Soil_Type6", "Soil_Type7", "Soil_Type8",
"Soil_Type9", "Soil_Type10", "Soil_Type11", "Soil_Type12", "Soil_Type13", "Soil_Type14", "Soil_Type15", "Soil_Type16", 
"Soil_Type17", "Soil_Type18", "Soil_Type19", "Soil_Type20", "Soil_Type21", "Soil_Type22", "Soil_Type23", "Soil_Type24", 
"Soil_Type25", "Soil_Type26", "Soil_Type27", "Soil_Type28", "Soil_Type29", "Soil_Type30", "Soil_Type31", "Soil_Type32", 
"Soil_Type33", "Soil_Type34", "Soil_Type35", "Soil_Type36", "Soil_Type37", "Soil_Type38", "Soil_Type39", "Soil_Type40"]
convertir_categoricas(data, lista)



# %% [markdown]
# ## Separando variables
# Es necesario separar los tipos de suelo, para analizar el dataframe sin tanto consumo de memoria, estas serán almacenadas en un nuevo dataframe, posteriormente podran unirse ya que mantienen el identificador de la fila a la que pertenecen.

# %%
lista.insert(0, "Id")
df_soil = data[lista]
df_soil.head(10)

# %%
lista.pop(0)
data.drop(lista, axis=1, inplace=True)
data.head()



# %% [markdown]
# ## Almacenando los dataset resultantes

# %%
data_file="../data/intermediate/forest-limpio.csv"
data.to_csv(data_file, index=False)
print(data_file, " almacenado.")
data_file="../data/intermediate/forest-limpio.pkl"
data.to_pickle(data_file)
print(data_file, " almacenado.")

# %%
data_file="../data/intermediate/forest-tipo-suelo.pkl"
df_soil.to_pickle(data_file)
print(data_file, " almacenado.")


# %% [markdown]
# # Limpieza de dataset Forest concluida.