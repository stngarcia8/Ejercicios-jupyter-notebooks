# %% [markdown]
# # Transformando dataset de Netflix


# %%
import pandas as pd


# %% [markdown]
# ## Cargando dataset limpio

# %%
data_file="../data/intermediate/netflix-limpio.csv"
data = pd.read_csv(data_file)
data.shape 


# %% [markdown]
# ## Visualizando variables a normalizar
# Las variables con la menor cantidad de valores unicos, son candidatas
# a dejarlas como clasificadoras, en este caso es el tipo y el publico.

# %%
def valores_unicos(dataframe):
    for column in dataframe:
        nombre_col = dataframe[column].name
        unicos_col = len(dataframe[column].unique())
        tipo_col = dataframe[column].dtype
        print("columna {}".format(nombre_col),
            "valores unicos {}".format(unicos_col),
            "tipo {}".format(tipo_col),
            sep="\t") 
valores_unicos(data)


# %% [markdown]
# ## Normalizando columna tipo

# %%
pelicula =['Movie']
tv_serie=['TV Show']
data.loc[data['tipo'].isin(pelicula), 
             'tipo'] = 'Pelicula'
data.loc[data['tipo'].isin(tv_serie), 
             'tipo'] = 'Serie'
data.head()


# %% [markdown]
# ## Normalizando columna publico
# Esta es la clasificacion del publico que tiene una pelicula
# o serie de television, siendo estas infantil, adolecente y adulto,

# %%
infantil=['G','TV-G','TV-Y','PG','TV-Y7', 'TV-Y7-FV', 'TV-PG']
adolecentes=['PG-13', 'TV-14']
adultos=['R', 'NC-17', 'TV-MA', 'NR', 'UR'] 
data.loc[data['publico'].isin(infantil), 'publico'] = 'infantil'
data.loc[data['publico'].isin(adolecentes), 'publico'] = 'Adolecente'
data.loc[data['publico'].isin(adultos), 'publico'] = 'Adulto'
data.head()


# %% [markdown]
# ## Contando valor de la columna tipo

# %%
data["tipo"] = data["tipo"].astype("category")
data["tipo"].dtype

# %%
data["tipo"].value_counts()


# %% [markdown]
# ## Contando valor de la columna publico

# %%
data["publico"] = data["publico"].astype("category")
data["publico"].dtype

# %%
data["publico"].value_counts()


# %% [markdown]
# ## Extrayendo los paises

# %%
data_copy = data.copy()
data_copy

# %%
def configura_dataframe(df, campo):
    split_data = df[campo].str.split(', ',expand=True)
    df.drop([campo], axis=1, inplace=True)
    df = pd.concat([df, split_data], axis=1)
    df = df.transpose()
    df.head(15)
    return df 

data_paises = data_copy[["id", "pais"]]
data_paises = configura_dataframe(data_paises, "pais")

# %%
def expandir_dataframe(df):
    lista=[]
    for col in df:
        for i in range(len(df[col].values)):
            if df[col].values[i] is not None and i>0:
                lista.append([df[col].values[0], df[col].values[i]])
    return lista

# %%
def grabar_dataframe(df, nombre_archivo):
    df.to_csv(nombre_archivo, index=False, header=False)
    print(nombre_archivo, " almacenado.")

data_paises=pd.DataFrame(expandir_dataframe(data_paises))
grabar_dataframe(data_paises, "../data/intermediate/netflix-paises.csv")


# %% [markdown]
# ## Extrayendo clasificaciones

# %%
data_clasificacion = data_copy[["id", "clasificacion"]]
data_clasificacion = configura_dataframe(data_clasificacion, "clasificacion")
data_clasificacion.head(15)
data_clasificacion=pd.DataFrame(expandir_dataframe(data_clasificacion))
grabar_dataframe(data_clasificacion, "../data/intermediate/netflix-clasificacion.csv")


# %% [markdown]
# ## Extrayendo actores

# %%
data_actores = data_copy[["id", "actores"]]
data_actores = configura_dataframe(data_actores, "actores")
data_actores.head(15)
data_actores=pd.DataFrame(expandir_dataframe(data_actores))
grabar_dataframe(data_actores, "../data/intermediate/netflix-actores.csv")


# %% [markdown]
# ## Extrayendo directores

# %%
data_directores = data_copy[["id", "director"]]
data_directores = configura_dataframe(data_directores, "director")
data_directores.head(15)
data_directores = pd.DataFrame(expandir_dataframe(data_directores))
grabar_dataframe(data_directores, "../data/intermediate/netflix-directores.csv")


# %% [markdown]
# ## Quitando columnas normalizadas
# Ya se ha descompuesto el dataset intermedio de netflix,
# Ahora las columnas, director, actores, paises y clasificacion
# no cuentasn para el proceso directamente deberan
# ser relacionadas directamente con los archivos generados
# "" por las diferentes normalizaciones que fueron realizadas.

# %%
data = data.drop(["director"], axis = 1)
data = data.drop(["actores"], axis = 1)
data = data.drop(["pais"], axis = 1)
data = data.drop(["clasificacion"], axis = 1)
data.head(10)


# %% [markdown]
# ## Almacenando la transformacion del dataset
# El formato del archivo es .pkl, este tipo de archivo
# almacena la estructura de los objetos, esto permite
# mantener todos las categorias almacenadas internamente
# por lo tanto, no debere realizar filtrados o cosas por el estilo
# nuevamente.

# %%
data_file="../data/intermediate/netflix-categorizado.pkl"
data.to_pickle(data_file)
print(data_file, " almacenado.")
data_file="../data/intermediate/netflix-categorizado.csv"
data.to_csv(data_file, index=False, header=True)
print(data_file, " almacenado.")


# %% [markdown]
# # Finalizo la transformacion del dataset