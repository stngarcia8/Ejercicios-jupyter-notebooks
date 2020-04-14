# %% [markdown]
# # Explorando y limpiando dataset netflix


# %%
import pandas as pd


# %% [markdown]
# ## Cargando dataset de origen

# %%
data_file="../data/original/netflix.csv"
data = pd.read_csv(data_file)
data.shape 


# %% [markdown]
# ## Cambiando titulos de columnas

# %%
data.columns=[
    'id', 'tipo', 'titulo', 
    'director', 'actores', 'pais', 
    'incorporada', 'lanzamiento', 'publico', 
    'duracion', 'clasificacion', 'descripcion']
data.head()


# %% [markdown]
# ## Verificando tipos de datos

# %%
data.dtypes


# %% [markdown]
# ## Quitando columnas innecesarias

# %% 
data = data.drop(["descripcion"], axis = 1)
data.head()



# %% [markdown]
# ## Buscando columnas nulas

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


# %% [markdown]
# ## Asignando valores a filas nula y verificando nuevamente

# %%
data['director'].fillna('Sin director', inplace=True)
data['actores'].fillna('Sin actores', inplace=True)
data['pais'].fillna('No clasificada', inplace=True)
data['incorporada'].fillna('No definido', inplace=True)
data['publico'].fillna('Sin clasificar', inplace=True)
data=data 
buscar_nulos(data)


# %% [markdown]
# ## Grabando el dataset limpio (intermedio)

# %%
data_file="../data/intermediate/netflix-limpio.csv"
data.to_csv(data_file, index=False)
print(data_file, " almacenado.")

# %% [markdown]
# # Limpieza de dataset netflix concluida.