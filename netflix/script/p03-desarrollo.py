# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (12,12)

# %% [markdown]
# ### Generando funciones de apoyo a los resultados

# %%
data_file="../data/intermediate/netflix-categorizado.pkl"
data_origen = pd.read_pickle(data_file)

# %%
def cargar_dataset(nombre_archivo, nombre_columna):
    df = pd.read_csv(nombre_archivo)
    df.columns=["id", nombre_columna]
    return df 

# %%
%matplotlib inline 
def graficar(titulo, xlabel, ylabel):
    plt.title(titulo)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

# %%
def combinar_dataframe(df_left, df_right):
    df_merged = pd.merge(
        left=df_left, right=df_right, 
        left_on='id', right_on='id')
    return df_merged




# %% [markdown]
# # Pregunta 1: Comprender qué contenido está disponible en diferentes países

# %%
data_paises =cargar_dataset("../data/intermediate/netflix-paises.csv", "pais")
paises_merged=combinar_dataframe(data_paises, data_origen)

# %%
copia = paises_merged.copy()
df =copia[["id", "publico"]]
df["cantidad"]=0
contenido = df["publico"].value_counts().to_frame()
contenido.reindex()
contenido["porcentaje"]=round(((contenido["publico"]/len(paises_merged))*100),1)

# %% [markdown]
# ## Tipos de contenidos brindados por Netflix

# %% [markdown]
# Netflix posee un amplio catálogo que diferencia el tipo de contenido a diferente publico, con la muestra obtenida se identifican las siguientes clasificaciones de público objetivo:

# %%
contenido

# %%
contenido = paises_merged.groupby("publico")["publico"].count().plot(kind="barh", legend="Reverse").legend(["Tipo de publico"])
graficar("Contenido publicado según tipo de público", "Cantidad de publicaciones", "Tipo de público")

# %% [markdown]
# La gráfica muestra como el contenido para adultos (mayores de 18 años) es el prevalente con un 45.1%.
# Por otra parte los adolecentes (entre 13 y 18 años) tienen un 30.8% de las publicaciones.
# El público infantil (menores a 8 años) tambien poseen contenido exclusivo para ellos y representa el 24%.
# Existen en la muestra items no etiquetados que tan solo son el 0.1% y fueron catalogados como "Sin clasificar".


# %% [markdown]
# Para este analisis serán utilizados los paises de America del sur debido a que la muestra en este aspecto es muy extensa.

# %%
paises_america =["Argentina", "Brazil", "Bolivia", "Chile", "Colombia", "Ecuador", "Paraguay", "Peru", "Uruguay", "Venezuela"]
df = paises_merged.loc[paises_merged["pais"].isin(paises_america), ["id", "pais", "publico"]]
print("Cantidad de titulos encontrados:", len(df))
contenido= df.pivot_table(index="pais", columns="publico", aggfunc="count", fill_value=0.0, margins=True, margins_name="Total")
contenido.head(10)

# %%
contenido= df.pivot_table(index="pais", columns="publico", aggfunc="count", fill_value=0)
contenido.plot(kind="barh", legend="Reverse").legend(["Adolecente", "Adulto", "Infantil"])
graficar("Relación de tipo de contenido publicado en America del sur", "Cantidad de publicaciones", "País")

# %% [markdown]
# La gráfica indica que en Argentina, Brasil, Chile y Colombia son los paises con mas contenido disponible, siendo Argentina el país con mas publicaciones para adultos (mayores de 18 años).
# Sin embargo, Colombia posee la mayoria de publicaciones para adolecentes (entre 13 y 17 años) y en Brasil se publica un contenido para niños (menores a 13 años) superior al resto de los paises.






# %% [markdown]
# # Pregunta 2: Identificar contenido similar haciendo coincidir características basadas en países

# %%
data_clasificacion =cargar_dataset("../data/intermediate/netflix-clasificacion.csv", "clasificacion")
clasificacion_merged=combinar_dataframe(data_clasificacion, data_origen)

# %%
copia = clasificacion_merged.copy()
df =copia[["id", "clasificacion"]]
df["cantidad"]=0
contenido = df["clasificacion"].value_counts().to_frame()
contenido.reindex()
contenido["porcentaje"]=round(((contenido["clasificacion"]/len(clasificacion_merged))*100),1)

# %% [markdown]
# Netflic posee muchas variedades de contenido siendo estas las siguientes:

# %%
contenido 

# %%
plt.rcParams['figure.figsize'] = (15,25)
clasificacion_merged.clasificacion.groupby(clasificacion_merged.clasificacion).count().plot(kind="pie")
graficar("Contenido publicado según clasificación", "", "")

# %% [markdown]
# Dentro de todas las clasificaciones encontradas, el mayor porcentaje de publicaciones pertenece a películas internacionales (14.1%), seguida por dramas (11.9%), comedias (8.1%) y  shows de televisión internacional (7.3%)
# A continuación se visualizarán estas cuatro clasificaciones en los principales paises de latinoamerica donde Netflix posee mayor cantidad de publicaciones que segun el punto 2, resultaron ser Argentina, Brasil, Chile y Colombia.

# %%
clasificacion_merged = combinar_dataframe(data_clasificacion, data_paises)
paises_america=["Argentina", "Brazil", "Chile", "Colombia"]
clasificaciones=["International Movies", "Dramas", "Comedies", "International TV Shows"]
df = clasificacion_merged.loc[clasificacion_merged["pais"].isin(paises_america), ["id", "pais", "clasificacion"]]
df = df.loc[df["clasificacion"].isin(clasificaciones), ["id", "pais", "clasificacion"]]
df.sort_values(["pais", "clasificacion"], ascending=True, inplace=True)
print("Cantidad de películas encontradas:", len(df))
contenido= df.pivot_table(index="pais", columns="clasificacion", aggfunc="count", fill_value=0, margins=True, margins_name="Total")
contenido.head()

# %%
plt.rcParams['figure.figsize'] = (12,12)
contenido= df.pivot_table(index="pais", columns="clasificacion", aggfunc="count", fill_value=0)
contenido.plot(kind="barh", legend="Reverse").legend(["Comedies", "Dramas","International Movies", "International TV Shows"])
graficar("Relacion de contenido entre paises y clasificaciones", "Cantidad de publicaciones", "País")

# %% [markdown]
# En argentina se han publicado la mayor cantidad de películas internacionales (42%) y dramas (27%), pero en Brasil los documentales y las comedias prevalecen con un 11% .
# Mientras tanto Chile y Colombia son los que menos contenido poseen en este tipo de clasificaciones.






# %% [markdown]
# # Pregunta 3: Análisis de la red de actores / directores y encontrar ideas interesantes en base a países y años

# %% [markdown]
# ## Analisis de directores

# %%
data_director =cargar_dataset("../data/intermediate/netflix-directores.csv", "director")
director_merged  = combinar_dataframe(data_director, data_origen)
copia = director_merged.copy()
df =copia[["id", "director", "lanzamiento"]]
contenido = df["director"].value_counts().to_frame()
contenido = contenido[1:11]

# %% [markdown]
# Los directores con mas películas publicadas en Netflix son:

# %%
contenido 

# %%
directores = contenido["director"].to_dict().keys()
copia = director_merged.copy()
df = copia.loc[copia["director"].isin(directores), ["id", "director", "lanzamiento"]]

# %% [markdown]
# Tabla de relación de año de lanzamiento versus director

# %%
contenido = df.pivot_table(index="lanzamiento", columns="director", aggfunc="count", fill_value=0, margins=True)
contenido 


# %% [markdown]
# ## Analisis de actores

# %%
data_actores =cargar_dataset("../data/intermediate/netflix-actores.csv", "actor")
actores_merged  = combinar_dataframe(data_actores, data_origen)
copia = actores_merged.copy()
df =copia[["id", "actor", "lanzamiento"]]
contenido = df["actor"].value_counts().to_frame()
contenido = contenido[1:11]

# %% [markdown]
# Los actores que han participado en más películas son:

# %%
contenido 

# %%
actores = contenido["actor"].to_dict().keys()
copia = actores_merged.copy()
df = copia.loc[copia["actor"].isin(actores), ["id", "actor", "publico"]]
print("Titulos encontrados:", len(df))
contenido = df.pivot_table(index="actor", columns="publico", aggfunc="count", fill_value=0, margins=True, margins_name="Total")
contenido 

# %%
contenido = df.pivot_table(index="actor", columns="publico", aggfunc="count", fill_value=0).plot(kind="barh", legend="Reversed").legend(["Adolecente", "Adulto", "Infantil"])
graficar("Relación entre actores con mayor participaciones y tipo de público", "Cantidad de películas", "Actores")






# %% [markdown]
# # Pregunta 4: ¿Netflix se ha centrado cada vez más en la televisión en lugar de las películas en los últimos años?

# %% [markdown]
# ## Relación de contenido de películas y series de TV.
# El desglose de tipo de contenido es el siguiente:

# %%
copia  = data_origen.copy()
df = copia[["id", "tipo"]]
contenido = df["tipo"].value_counts().to_frame()
contenido.reindex()
contenido["porcentaje"]=round(((contenido["tipo"]/len(data_origen))*100),1)
contenido 

# %%
plt.rcParams['figure.figsize'] = (12,12)
data_origen.tipo.groupby(data_origen.tipo).count().plot(kind="pie")
graficar("Relación entre Peliculas vs Series de televisión", "", "")

# %% [markdown]
# La muestra posee 6234 registros, de los cuales, 4265 (68.4%) corresponden a películas, mientras tanto, 1969 son series de televisión que es un 31.6% de la muestra.



# %% [markdown]
# ## Relación de contenido de películas y series de TV años 2017-2020

# %%
last_years=data_origen.loc[data_origen["lanzamiento"]>=2017, ["lanzamiento", "tipo"]]
contenido = last_years["tipo"].value_counts().to_frame()
contenido.reindex()
contenido["porcentaje"]=round(((contenido["tipo"]/len(last_years))*100),1)

# %% [markdown]
# Entre los años 2017 y 2020 Netflix ha publicado el siguiente contenido:

# %%
contenido 

# %%
last_years.lanzamiento.groupby(last_years.tipo).count().plot(kind="pie")
graficar("Relación entre Peliculas vs Series de televisión (2017 - 2020)", "", "")

# %% [markdown]
# Las publicaciones durante el periodo 2017 y 2020 fueron de 2890 elementos, de los cuales el 60% fueron peliculas (1734) y el 40% son series de televisión (1156)

# %% [markdown]
# ## Relación de contenido de películas y series de TV de los últimos cuatro años (2017-2020)

# %%
fig=plt.figure(figsize=(15,15))
for i in range(2017,2021):
    ax = plt.subplot(i-1796).set_title("Año "+ str(i).strip())
    df=last_years.loc[last_years["lanzamiento"]==i, ["lanzamiento", "tipo"]]
    df.lanzamiento.groupby(df.tipo).count().plot(kind="pie")

# %%
%matplotlib inline 
plt.show()


# %% [markdown]
# En los últimos cuatro años, Netflix ha publicado mas películas que series de televisión, en la gráfica se visualiza que en los años 2019 y 2020 el tipo de contenido publicado de series de televisión es superior a las películas, contrastando los resultados de los años 2017 y 2018 en donde las películas poseen la mayoría de publicaciones.

