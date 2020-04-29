# %% [markdown]
# # Transformando dataset de Forest


# %%
import pandas as pd
import matplotlib.pyplot as plt


# %% [markdown]
# ## Cargando dataset limpio

# %%
data_file="../data/intermediate/forest-limpio.csv"
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
# ## Variables categoricas identificadas
# Según el resultado, las variables categoricas que se encuentran en el dataframe corresponden a:
# Cover_Type
# Wilderness_Area
# Soil_Type



# %% [markdown]
# ### Wildernes_Area
# Corresponde a las áreas silvestres con menos perturbaciones realizadas por el hombre, la covertura es natural y no posee intervenciones forestales.
# Posee valores numéricos comprendidos entre 1 y 4

# %%
data.loc[data["Wilderness_Area"]=="Wilderness1", "Wilderness_Area"]=1
data.loc[data["Wilderness_Area"]=="Wilderness2", "Wilderness_Area"]=2
data.loc[data["Wilderness_Area"]=="Wilderness3", "Wilderness_Area"]=3
data.loc[data["Wilderness_Area"]=="Wilderness4", "Wilderness_Area"]=4
data["Wilderness_Area"] = data["Wilderness_Area"].astype("category")
data["Wilderness_Area"].value_counts()

# %%
%matplotlib inline 
df_wilderness = data.groupby(data["Wilderness_Area"])["Wilderness_Area"].count().plot(kind="barh", legend="Reverse")
plt.title("Distribución de áreas salvajes")
plt.ylabel("Tipo de area silvestre")
plt.xlabel("Nro. ocurrencias")
plt.show()

# %% [markdown]
# El gráfico muestra la distribución de las áreas silvestres y la cantidad obtenida es igual al total de instancias observadas, por lo cual esta variable categórica es normalizada correctamente.



# %% [markdown]
# ### Cover_Type
# Es el tipo de covertura del bosque, posee valores numéricos con valores desde 1 hasta 7

# %%
data["Cover_Type"] = data["Cover_Type"].astype("category")
data["Cover_Type"].value_counts()

# %%
%matplotlib inline 
df_cover = data.groupby(data["Cover_Type"])["Cover_Type"].count().plot(kind="barh", legend="Reverse")
plt.title("Distribución de tipos de covertura")
plt.ylabel("Tipo de covertura")
plt.xlabel("Nro. ocurrencias")
plt.show()

# %% [markdown]
# La gráfica muestra que los valores estan distribuidos uniformemente en la variable covertura, la cantidad es  correcta, pero tiene diferencias sustanciales en relacion a la altura y esto puede satisfacer ciertas preguntas que en el desarrollo pueden servir para conocimiento de las especies encontradas en los bosques, por tal motivo, esta variable categórica es tomada en cuenta en el dataset.



# %% [markdown]
# ### SoilType
# Es el tipo de suelo encontrado en el bosque y comprenden valores entre 1 y 40, según "USFS Ecological unit Landtype"

# %%
for i in range(1,40):
    columna = "Soil"+str(i)
    data.loc[data["Soil_Type"]==columna, "Soil_Type"]=i


# %%
data["Soil_Type"].value_counts()

# %%
%matplotlib inline 
df_cover = data.groupby(data["Soil_Type"])["Soil_Type"].count().plot(kind="bar", legend="Reverse")
plt.title("Distribución de tipos de suelo")
plt.xlabel("Tipo de suelo")
plt.ylabel("Nro. ocurrencias")
plt.show()

# %% [markdown]
# Los tipos de suelo se encuentran correctamente normalizados, el único tipo de suelo que no aparece en la muestra es el tipo 15.



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