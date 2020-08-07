# %% [markdown]
# # Evaluación 02 -  Juegos Olímpicos
# Alumno: Daniel García Loyola<br>
# Profesor: Israel Naranjo Retamal<br>
# Ramo: Minería de datos (MDY7101)<br>
# Sección: 002D<br>
# Entrega: 12/06/2020<br>
# Pregunta: 01


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
plt.rcParams['figure.figsize'] = (8, 8)


# %% [markdown]
# # Cargando datos de juegos olímpicos

# %%
data_file = "../data/original/JuegosOlimpicos.csv"
data = pd.read_csv(data_file, engine='python')
data.shape

# %% [markdown]
# Ajustando nombres de columnas.

# %%
data.columns = [
    'ID', 'NombreAtleta', 'Sexo',
    'Edad', 'Altura', 'Peso',
    'Equipo', 'Comite',
    'JuegosOlimpicos', 'Año', 'Temporada', 'Ciudad',
    'Deporte', 'Competencia', 'Medalla']
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
# Para el estudio son requeridos las observaciones que poseen medallas,
# por lo tanto, tan solo estas observaciones serán tomadas en cuenta
# los atletas que no hayan ganado medallas serán eliminados del dataframe.

# %%
print("Cantidad de elementos nulos por columna:")
data.isnull().sum()

# %%
data.dropna(inplace=True)
print("Resultado luego de eliminar nulos: ")
data.isnull().sum()


# %% [markdown]
# ## Transformando variables
# Cambiar ciertas columnas para un mejor tratamiento de los datos relacionados.

# %%
print("Cambiando columna sexo a valores numéricos")
data['Sexo_temp'] = 0
data.loc[data.Sexo == 'M', 'Sexo_temp'] = 1
data.loc[data.Sexo == 'F', 'Sexo_temp'] = 0


# %%
print("Cambiando columna medalla a valores numéricos")
data['Medalla_temp'] = 0
data.loc[data.Medalla == 'Gold', 'Medalla_temp'] = 1
data.loc[data.Medalla == 'Silver', 'Medalla_temp'] = 2
data.loc[data.Medalla == 'Bronze', 'Medalla_temp'] = 3


# %%
data.head(15)

# %% [markdown]
# ## Eliminar columnas inecesarias en el dataframe

# %%
columnas = [
    'ID', 'NombreAtleta', 'Sexo',
    'JuegosOlimpicos', 'Temporada', 'Ciudad',
    'Año', 'Equipo', 'Comite', 'Competencia']
data = data.drop(columnas, axis=1)
data.head()

# %%
print("Restableciendo nombres de columnas")
data.columns = [
    'Edad', 'Altura', 'Peso',
    'Deporte', 'Medalla', 'Sexo', 'MedallaNum']
data.columns


# %% [markdown]
# # Visualizando correlación de las columnas

# %%
data_corr = data.corr(method='pearson')
sns.set_style(style='white')
fig, ax = plt.subplots(figsize=(5, 3))
sns.heatmap(data_corr, ax=ax)
plt.title("Correlación de las variables del dataframe")


# %%
data_corr

# %% [markdown]
# Se puede ver que existe una alta correlación entre altura y peso,
# la columna sexo, posee una debil correlacion con las mencionadas anteriormente,
# las demás columnas no poseen relación alguna entre si.
# <hr style="border:2px; width:75%;">

# %% [markdown]
# # Verificando distribución de las columnas
# Se requere verificar la distribución para poder saber si es que debe ser aplicado
# algún metodo de standarización a los datos, las variables interesantes
# para este estudio corresponden a la edad, altura ypeso de los atletas,
# por lo tanto, las correcciones serán realizadas a dichas columnas, las demás pueden que sirvan de predictores solamente.


# %%
from scipy import stats


# %%
plt.figure(figsize=(10, 3))
plt.title('Relación de simetria de las columnas edad, peso y altura')
plt.subplot(1, 3, 1)
data.Edad.plot.kde()
plt.subplot(1, 3, 2)
data.Peso.plot.kde()
plt.subplot(1, 3, 3)
data.Altura.plot.kde()
plt.show()

# %%
print('Coeficientes de simetría: ')
print('Edad    : ', stats.skew(data.Edad), sep='\t')
print('Peso    : ', stats.skew(data.Peso), sep='\t')
print('Altura : ', stats.skew(data.Altura), sep='\t')


# %% [markdown]
# Vemos que las columnas estan mas cargadas al costado derecho del gráfico,
# será necesario realizar la eliminación de extremos para poder
# normalizar las columnas y asi los modelos predictivos a aplicar
# no se sesgaran por los valores mas altos que encuentren.

# %%
def quitar_extremos(df):
    cantidad_outliers = 0
    for columna in df:
        if df[columna].dtype != np.object:
            n_outliers = len(df[np.abs(stats.zscore(df[columna])) > 3])
            print("{} {} {}".format(
                df[columna].name, n_outliers, df[columna].dtype))
            if n_outliers > 0:
                cantidad_outliers = n_outliers
    return True if cantidad_outliers > 0 else False


# %%
while quitar_extremos(data[['Edad', 'Peso', 'Altura']]):
    data = data[
        (np.abs(stats.zscore(data.Edad)) < 3)  &
        (np.abs(stats.zscore(data.Peso)) < 3) &
        (np.abs(stats.zscore(data.Altura)) < 3)]


# %% [markdown]
# Ahora vuelvo a visualizar para verificar que los extremos fueron eliminados correctamente.


# %%
plt.figure(figsize=(10, 3))
plt.title('Relación de simetria de las columnas edad, peso y altura')
plt.subplot(1, 3, 1)
data.Edad.plot.kde()
plt.subplot(1, 3, 2)
data.Peso.plot.kde()
plt.subplot(1, 3, 3)
data.Altura.plot.kde()
plt.show()

# %%
print('Coeficientes de simetría: ')
print('Edad    : ', stats.skew(data.Edad), sep='\t')
print('Peso    : ', stats.skew(data.Peso), sep='\t')
print('Altura : ', stats.skew(data.Altura), sep='\t')

# %% [markdown]
# Ahora las columnas de edad, peso y altura estan normalizadas,
# esto permitirá que los modelos sean mas parejos para realizar las
# preddicciones, recordar que deben ser eliminados los extremos
# (outliers) antes de proseguir con el ejercicio.


# %%[markdown]
# # Visualizando los tipos de medallas
# Debemos conocer el estado de la dictribución de los tipos de medallas
# para saber que tal será el desempeño del atleta a mostrar.


# %%
plt.figure(figsize=(6, 2))
data.groupby('Medalla')['Medalla'].value_counts(normalize=True).plot(kind='barh', legend='best', color='orange')
plt.title('Distribución de tipos de medallas')
plt.xlabel('Porcentaje')
plt.ylabel('Tipo medalla')
plt.show()

# %%
print('Tipos de medallas')
data.Medalla.value_counts()

# %% {markdown}
# ## Medallas de oro

# %%
df_oro = data.loc[data["Medalla"] == 'Gold', ]
while quitar_extremos(df_oro[['Edad', 'Peso', 'Altura']]):
    df_oro = df_oro[
        (np.abs(stats.zscore(df_oro.Edad)) < 3)  &
        (np.abs(stats.zscore(df_oro.Peso)) < 3) &
        (np.abs(stats.zscore(df_oro.Altura)) < 3)
        ]


# %%
def generar_grafico(df, columna, color):
    grafico = sns.distplot(
        df[columna],
        rug=True, rug_kws={"color": color},
        kde_kws={"color": "k", "lw": 3, "label": columna},
        hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": color})
    return grafico


# %%
plt.figure(figsize=(10, 3))
plt.subplot(1, 3, 1)
plt.title('Medallas de oro')
generar_grafico(df_oro, 'Edad', 'g')
plt.subplot(1, 3, 2)
generar_grafico(df_oro, 'Peso', 'r')
plt.title('Medallas de plata')
plt.subplot(1, 3, 3)
generar_grafico(df_oro, 'Altura', 'b')
plt.title('Medallas de bronce')
plt.show()


# %% {markdown}
# ## Medallas de plata


# %%
df_plata = data.loc[data["Medalla"] == 'Silver', ]
while quitar_extremos(df_plata[['Edad', 'Peso', 'Altura']]):
    df_plata = df_plata[
        (np.abs(stats.zscore(df_plata.Edad)) < 3)  &
        (np.abs(stats.zscore(df_plata.Peso)) < 3) &
        (np.abs(stats.zscore(df_plata.Altura)) < 3)
        ]


# %%
plt.figure(figsize=(10, 3))
plt.subplot(1, 3, 1)
plt.title('Medallas de oro')
generar_grafico(df_plata, 'Edad', 'g')
plt.subplot(1, 3, 2)
generar_grafico(df_plata, 'Peso', 'r')
plt.title('Medallas de plata')
plt.subplot(1, 3, 3)
generar_grafico(df_plata, 'Altura', 'b')
plt.title('Medallas de bronce')
plt.show()


# %% {markdown}
# ## Medallas de bronze


# %%
df_bronce = data.loc[data["Medalla"] == 'Bronze', ]
while quitar_extremos(df_bronce[['Edad', 'Peso', 'Altura']]):
    df_bronce = df_bronce[
        (np.abs(stats.zscore(df_bronce.Edad)) < 3)  &
        (np.abs(stats.zscore(df_bronce.Peso)) < 3) &
        (np.abs(stats.zscore(df_bronce.Altura)) < 3)
        ]


# %%
plt.figure(figsize=(10, 3))
plt.subplot(1, 3, 1)
plt.title('Medallas de oro')
generar_grafico(df_bronce, 'Edad', 'g')
plt.subplot(1, 3, 2)
generar_grafico(df_bronce, 'Peso', 'r')
plt.title('Medallas de plata')
plt.subplot(1, 3, 3)
generar_grafico(df_bronce, 'Altura', 'b')
plt.title('Medallas de bronce')
plt.show()


# %% [markdown]
# Luego de visualizar, procesar y explorar los datos,
# estamos en condiciones de poder aplicar los modelos
# necesarios para el ejercicio.


# %% [markdown]
# # Calcular las probabilidades que posee un atleta
# Para el ejercicio, es necesario aplicar diferentes
# modelos que nos permitan ingresar la edad, peso, altura
# sexo y deporte para poder visualizar si el atleta tendrá
#  posibilidades de exito en los próximos juegos olímpicos de Tokio 2020.


# %%
def obtener_dataframe(df, sexo, deporte, medalla):
    deporte_atleta = data.loc[:, 'Deporte'].str.contains(deporte, case=False)
    sexo_atleta = data.loc[:, 'Sexo'] == sexo
    if not deporte:
        df = df.loc[sexo_atleta]
    elif not sexo:
        df = df.loc[deporte_atleta]
    elif not sexo and not deporte:
        df = df
    else:
        df = df.loc[deporte_atleta & sexo_atleta]
    if medalla:
        medalla_atleta = data.loc[:, 'Medalla'] == medalla
        df = df.loc[medalla_atleta]
    return df


# %% [markdown]
# Los datos a utilizar para este ejercicio son los de la atleta **Kristel Koubrick**


# %%
edad = 34
peso = 61
altura = 161
sexo = 0
deporte = 'Swimming'
data_new = obtener_dataframe(data, sexo, deporte, '').drop_duplicates()
data_new = data_new.drop(['Medalla'], axis=1)
data_new = data_new.sort_values(['Edad', 'Altura', 'Peso'])
data_new

# %%
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
resultados = []
resultados_oro = []
resultados_plata = []
resultados_bronce = []


# %% [markdown]
# Revisar si las columnas son aptas para el modelo

# %%
def validar_columnas(df, tipo, X_columnas, y_columnas):
    X = df[X_columnas]
    Y = df[y_columnas]
    estimador = SVR(kernel=tipo)
    seleccionador = RFE(estimador, len(X), step=1)
    seleccionador = seleccionador.fit(X, Y)
    print(seleccionador.support_)
    return


# %% [markdown]
#  Generar los conjuntos de entrenamiento y test


# %%
def generar_entrenamiento(df, X_columnas, y_columnas):
    X = np.array(df[X_columnas])
    y = np.array(df[[y_columnas]].values)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    return X_train, X_test, y_train, y_test


# %% [markdown]
# ## Regresión lineal


# %% [markdown]
# ### Probabilidad general de obtener medallas


# %%
def mostrar_variables(interseccion, coeficiente, score, error_cuadratico):
    print()
    print('Intersección     :', interseccion, sep='\t')
    print('Coeficiente      :', coeficiente, sep='\t')
    print('Score            :', score, sep='\t')
    print('"Error cuadrático :', error_cuadratico, sep='\t')
    return

# %%
X_columnas = ['Edad', 'Peso', 'Altura']
y_columnas = 'MedallaNum'
validar_columnas(data_new, 'linear', X_columnas, y_columnas)
X_train, X_test, y_train, y_test = generar_entrenamiento(data_new, X_columnas, y_columnas)
lm = LinearRegression()
lm.fit(X_train, y_train)
y_pred = lm.predict(X_train)
interseccion = lm.intercept_[0]
coeficiente = lm.coef_[0][0]
score = lm.score(X_train, y_train)
error_cuadratico = mean_squared_error(y_train, y_pred)
mostrar_variables(interseccion, coeficiente, score, error_cuadratico)


# %% [markdown]
# Aplicar la formula a los elementos del dataframe


# %%
def aplicar_formula(df, coeficiente, interseccion):
    df['re'] = df[['Edad']].apply(lambda x: (coeficiente*x.Edad+interseccion), axis=1)
    df['rp'] = df[['Peso']].apply(lambda x: (coeficiente*x.Peso+interseccion), axis=1)
    df['ra'] = df[['Altura']].apply(lambda x: (coeficiente*x.Altura+interseccion), axis=1)
    return df
data_new = aplicar_formula(data_new, coeficiente, interseccion)
data_new.head(5)


# %% [markdown]
# #### Graficar resultados aplicados al dataframe


# %%
def graficar_resultados(df):
    plt.figure(figsize=(12, 3))
    plt.subplot(1, 3, 1)
    sns.scatterplot(df.Edad, df.re)
    plt.subplot(1, 3, 2)
    sns.scatterplot(df.Peso, df.rp)
    plt.subplot(1, 3, 3)
    sns.scatterplot(df.Altura, df.ra)
    plt.show()
    return
graficar_resultados(data_new)


# %% [markdown]
# #### Predecir por las características del atleta


# %%
y_pred = lm.predict(np.array([[edad, peso, altura]]))
prediccion = y_pred[0][0]
print('La probabilidad de ganar medalla es : {:.3f}'.format(prediccion))
resultados.append(['Regresión lineal', score, interseccion, coeficiente, error_cuadratico, prediccion])


# %% [markdown]
# ### Probabilidad medalla de oro


# %% [markdown]
# Obtener datos para medalla de **oro**


# %%
import random
data_oro = obtener_dataframe(data, sexo, deporte, 'Gold').drop_duplicates()
data_oro = data_oro.drop(['Sexo', 'Medalla', 'Deporte'], axis=1)
data_oro.MedallaNum = data_oro[['MedallaNum']].apply(lambda x : random.randrange(5), axis=1)
data_oro = data_oro.sort_values(['Edad', 'Altura', 'Peso'])
data_oro.head(5)

# %% [markdown]
# Realizar las predicciones


# %%
validar_columnas(data_oro, 'linear', X_columnas, y_columnas)
X_train, X_test, y_train, y_test = generar_entrenamiento(data_oro, X_columnas, y_columnas)
lm = LinearRegression()
lm.fit(X_train, y_train)
y_pred = lm.predict(X_train)
interseccion = lm.intercept_[0]
coeficiente = lm.coef_[0][0]
score = lm.score(X_train, y_train)
error_cuadratico = mean_squared_error(y_train, y_pred)
mostrar_variables(interseccion, coeficiente, score, error_cuadratico)


# %% [markdown]
# Aplicar la formula a los elementos del dataframe


# %%
data_oro = aplicar_formula(data_oro, coeficiente, interseccion)
data_oro.head(5)


# %% [markdown]
# #### Graficar resultados aplicados al dataframe


# %%
graficar_resultados(data_oro)


# %% [markdown]
# #### Predecir por las características del atleta


# %%
y_pred = lm.predict(np.array([[edad, peso, altura]]))
prediccion = y_pred[0][0]
print('La probabilidad de ganar medalla de oro es : {:.3f}'.format(prediccion))
resultados_oro.append(['Regresión lineal', score, interseccion, coeficiente, error_cuadratico, prediccion])


# %% [markdown]
# ### Probabilidad medalla de **plata**


# %% [markdown]
# Obtener datos para medalla de plata


# %%
data_plata = obtener_dataframe(data, sexo, deporte, 'Silver').drop_duplicates()
data_plata = data_plata.drop(['Sexo', 'Medalla', 'Deporte'], axis=1)
data_plata.MedallaNum = data_plata[['MedallaNum']].apply(lambda x : random.randrange(4), axis=1)
data_plata = data_plata.sort_values(['Edad', 'Altura', 'Peso'])
data_plata.head(5)

# %% [markdown]
# Realizar las predicciones


# %%
validar_columnas(data_plata, 'linear', X_columnas, y_columnas)
X_train, X_test, y_train, y_test = generar_entrenamiento(data_plata, X_columnas, y_columnas)
lm = LinearRegression()
lm.fit(X_train, y_train)
y_pred = lm.predict(X_train)
interseccion = lm.intercept_[0]
coeficiente = lm.coef_[0][0]
score = lm.score(X_train, y_train)
error_cuadratico = mean_squared_error(y_train, y_pred)
mostrar_variables(interseccion, coeficiente, score, error_cuadratico)


# %% [markdown]
# Aplicar la formula a los elementos del dataframe


# %%
data_plata = aplicar_formula(data_plata, coeficiente, interseccion)
data_plata.head(5)


# %% [markdown]
# #### Graficar resultados aplicados al dataframe


# %%
graficar_resultados(data_plata)


# %% [markdown]
# #### Predecir por las características del atleta


# %%
y_pred = lm.predict(np.array([[edad, peso, altura]]))
prediccion = y_pred[0][0]
print('La probabilidad de ganar medalla de plata es : {:.3f}'.format(prediccion))
resultados_plata.append(['Regresión lineal', score, interseccion, coeficiente, error_cuadratico, prediccion])


# %% [markdown]
# ### Probabilidad medalla de **bronce**


# %% [markdown]
# Obtener datos para medalla de bronce


# %%
data_bronce = obtener_dataframe(data, sexo, deporte, 'Bronze').drop_duplicates()
data_bronce = data_bronce.drop(['Sexo', 'Medalla', 'Deporte'], axis=1)
data_bronce.MedallaNum = data_bronce[['MedallaNum']].apply(lambda x : random.randrange(3), axis=1)
data_bronce = data_bronce.sort_values(['Edad', 'Altura', 'Peso'])
data_bronce.head(5)

# %% [markdown]
# Realizar las predicciones


# %%
validar_columnas(data_bronce, 'linear', X_columnas, y_columnas)
X_train, X_test, y_train, y_test = generar_entrenamiento(data_bronce, X_columnas, y_columnas)
lm = LinearRegression()
lm.fit(X_train, y_train)
y_pred = lm.predict(X_train)
interseccion = lm.intercept_[0]
coeficiente = lm.coef_[0][0]
score = lm.score(X_train, y_train)
error_cuadratico = mean_squared_error(y_train, y_pred)
mostrar_variables(interseccion, coeficiente, score, error_cuadratico)


# %% [markdown]
# Aplicar la formula a los elementos del dataframe


# %%
data_bronce = aplicar_formula(data_bronce, coeficiente, interseccion)
data_bronce.head(5)


# %% [markdown]
# #### Graficar resultados aplicados al dataframe


# %%
graficar_resultados(data_bronce)


# %% [markdown]
# #### Predecir por las características del atleta


# %%
y_pred = lm.predict(np.array([[edad, peso, altura]]))
prediccion = y_pred[0][0]
print('La probabilidad de ganar medalla de bronce es : {:.3f}'.format(prediccion))
resultados_bronce.append(['Regresión lineal', score, interseccion, coeficiente, error_cuadratico, prediccion])



# %% [markdown]
# ## Regresión logistica


# %% [markdown]
# ### Probabilidad general de obtener medallas# %% [markdown]


# %%
from sklearn.linear_model import LogisticRegression 
from sklearn.preprocessing import StandardScaler 
from sklearn import metrics


# %%
X_columnas = ['Edad', 'Peso', 'Altura']
y_columnas = 'MedallaNum'
X_train, X_test, y_train, y_test = generar_entrenamiento(data_new, X_columnas, y_columnas)
escalar = StandardScaler() 
X_train = escalar.fit_transform(X_train) 
X_test = escalar.transform(X_test) 
lm = LogisticRegression()
lm.fit(X_train, y_train)
y_pred = lm.predict(X_train)
interseccion = lm.intercept_[0]
coeficiente = lm.coef_[0][0]
score = lm.score(X_train, y_train)
error_cuadratico = mean_squared_error(y_train, y_pred)
mostrar_variables(interseccion, coeficiente, score, error_cuadratico)

# %% [markdown]
# #### Predecir por las características del atleta


# %%
y_pred = lm.predict(np.array([[edad, peso, altura]]))
prediccion = y_pred[0] 
print('La probabilidad de ganar medalla es : {:.3f}'.format(prediccion))
resultados.append(['Regresión logistica', score, interseccion, coeficiente, error_cuadratico, prediccion])


# %% [markdown]
# ### Probabilidad de obtener medalla de **oro**


# %%
X_train, X_test, y_train, y_test = generar_entrenamiento(data_oro, X_columnas, y_columnas)
escalar = StandardScaler() 
X_train = escalar.fit_transform(X_train) 
X_test = escalar.transform(X_test) 
lm = LogisticRegression()
lm.fit(X_train, y_train)
y_pred = lm.predict(X_train)
interseccion = lm.intercept_[0]
coeficiente = lm.coef_[0][0]
score = lm.score(X_train, y_train)
error_cuadratico = mean_squared_error(y_train, y_pred)
mostrar_variables(interseccion, coeficiente, score, error_cuadratico)

# %% [markdown]
# #### Predecir por las características del atleta


# %%
y_pred = lm.predict(np.array([[edad, peso, altura]]))
prediccion = y_pred[0]
print('La probabilidad de ganar medalla de oro es : {:.3f}'.format(prediccion))
resultados_oro.append(['Regresión logistica', score, interseccion, coeficiente, error_cuadratico, prediccion])


# %% [markdown]
# ### Probabilidad de obtener medalla de **plata**


# %%
X_train, X_test, y_train, y_test = generar_entrenamiento(data_plata, X_columnas, y_columnas)
escalar = StandardScaler() 
X_train = escalar.fit_transform(X_train) 
X_test = escalar.transform(X_test) 
lm = LogisticRegression()
lm.fit(X_train, y_train)
y_pred = lm.predict(X_train)
interseccion = lm.intercept_[0]
coeficiente = lm.coef_[0][0]
score = lm.score(X_train, y_train)
error_cuadratico = mean_squared_error(y_train, y_pred)
mostrar_variables(interseccion, coeficiente, score, error_cuadratico)

# %% [markdown]
# #### Predecir por las características del atleta


# %%
y_pred = lm.predict(np.array([[edad, peso, altura]]))
prediccion = y_pred[0]
print('La probabilidad de ganar medalla de plata es : {:.3f}'.format(prediccion))
resultados_plata.append(['Regresión logistica', score, interseccion, coeficiente, error_cuadratico, prediccion])


# %% [markdown]
# ### Probabilidad de obtener medalla de **bronce**


# %%
X_train, X_test, y_train, y_test = generar_entrenamiento(data_bronce, X_columnas, y_columnas)
escalar = StandardScaler() 
X_train = escalar.fit_transform(X_train) 
X_test = escalar.transform(X_test) 
lm = LogisticRegression()
lm.fit(X_train, y_train)
y_pred = lm.predict(X_train)
interseccion = lm.intercept_[0]
coeficiente = lm.coef_[0][0]
score = lm.score(X_train, y_train)
error_cuadratico = mean_squared_error(y_train, y_pred)
mostrar_variables(interseccion, coeficiente, score, error_cuadratico)

# %% [markdown]
# #### Predecir por las características del atleta


# %%
y_pred = lm.predict(np.array([[edad, peso, altura]]))
prediccion = y_pred[0]
print('La probabilidad de ganar medalla de bronce es : {:.3f}'.format(prediccion))
resultados_bronce.append(['Regresión logistica', score, interseccion, coeficiente, error_cuadratico, prediccion])



# %% [markdown]
# ## Vector categorizado


# %% [markdown]
# ### Probabilidad general de obtener medallas# %% [markdown]


# %%
from sklearn.svm import SVC



# %%
X_columnas = ['Edad', 'Peso', 'Altura']
y_columnas = 'MedallaNum'
X_train, X_test, y_train, y_test = generar_entrenamiento(data_new, X_columnas, y_columnas)
svc = SVC()
svc.fit(X_train, y_train)
y_pred = svc.predict(X_train)
interseccion = svc.intercept_[0]
score = svc.score(X_train, y_train)
error_cuadratico = mean_squared_error(y_train, y_pred)
mostrar_variables(interseccion, 0, score, error_cuadratico)


# %% [markdown]
# #### Predecir por las características del atleta


# %%
y_pred = svc.predict(np.array([[edad, peso, altura]]))
prediccion = y_pred[0] 
print('La probabilidad de ganar medalla es : {:.3f}'.format(prediccion))
resultados.append(['Vector supervisado (SVC)', score, interseccion, coeficiente, error_cuadratico, prediccion])


# %% [markdown]
# ### Probabilidad de obtener medalla de **oro**


# %%
X_train, X_test, y_train, y_test = generar_entrenamiento(data_oro, X_columnas, y_columnas)
svc = SVC()
svc.fit(X_train, y_train)
y_pred = svc.predict(X_train)
interseccion = svc.intercept_[0]
score = svc.score(X_train, y_train)
error_cuadratico = mean_squared_error(y_train, y_pred)
mostrar_variables(interseccion, 0, score, error_cuadratico)


# %% [markdown]
# #### Predecir por las características del atleta


# %%
y_pred = svc.predict(np.array([[edad, peso, altura]]))
prediccion = y_pred[0] 
print('La probabilidad de ganar medalla de oro es : {:.3f}'.format(prediccion))
resultados_oro.append(['Vector supervisado (SVC)', score, interseccion, coeficiente, error_cuadratico, prediccion])



# %% [markdown]
# ### Probabilidad de obtener medalla de **plata**


# %%
X_train, X_test, y_train, y_test = generar_entrenamiento(data_plata, X_columnas, y_columnas)
svc = SVC()
svc.fit(X_train, y_train)
y_pred = svc.predict(X_train)
interseccion = svc.intercept_[0]
score = svc.score(X_train, y_train)
error_cuadratico = mean_squared_error(y_train, y_pred)
mostrar_variables(interseccion, 0, score, error_cuadratico)


# %% [markdown]
# #### Predecir por las características del atleta


# %%
y_pred = svc.predict(np.array([[edad, peso, altura]]))
prediccion = y_pred[0] 
print('La probabilidad de ganar medalla de plata es : {:.3f}'.format(prediccion))
resultados_plata.append(['Vector supervisado (SVC)', score, interseccion, coeficiente, error_cuadratico, prediccion])


# %% [markdown]
# ### Probabilidad de obtener medalla de **bronce**


# %%
X_train, X_test, y_train, y_test = generar_entrenamiento(data_bronce, X_columnas, y_columnas)
svc = SVC()
svc.fit(X_train, y_train)
y_pred = svc.predict(X_train)
interseccion = svc.intercept_[0]
score = svc.score(X_train, y_train)
error_cuadratico = mean_squared_error(y_train, y_pred)
mostrar_variables(interseccion, 0, score, error_cuadratico)


# %% [markdown]
# #### Predecir por las características del atleta


# %%
y_pred = svc.predict(np.array([[edad, peso, altura]]))
prediccion = y_pred[0] 
print('La probabilidad de ganar medalla de bronce es : {:.3f}'.format(prediccion))
resultados_bronce.append(['Vector supervisado (SVC)', score, interseccion, coeficiente, error_cuadratico, prediccion])


# %% [markdown]
# ## Arbol de desición


# %% [markdown]
# ### Probabilidad general de obtener medallas# %% [markdown]


# %%
from sklearn.tree import DecisionTreeClassifier


# %%
X_columnas = ['Edad', 'Peso', 'Altura']
y_columnas = 'MedallaNum'
X_train, X_test, y_train, y_test = generar_entrenamiento(data_new, X_columnas, y_columnas)
dtc = DecisionTreeClassifier()
dtc.fit(X_train, y_train)
y_pred = dtc.predict(X_train)
score = dtc.score(X_train, y_train)
error_cuadratico = mean_squared_error(y_train, y_pred)
mostrar_variables(0, 0, score, error_cuadratico)


# %% [markdown]
# #### Predecir por las características del atleta



# %%
y_pred = dtc.predict(np.array([[edad, peso, altura]]))
prediccion = y_pred[0] 
print('La probabilidad de ganar medalla es : {:.3f}'.format(prediccion))
resultados.append(['Arvol de desición', score, 0, 0, error_cuadratico, prediccion])








# %% [markdown]
# ### Probabilidad de ganar medalla de **oro**


# %%
X_train, X_test, y_train, y_test = generar_entrenamiento(data_oro, X_columnas, y_columnas)
dtc = DecisionTreeClassifier()
dtc.fit(X_train, y_train)
y_pred = dtc.predict(X_train)
score = dtc.score(X_train, y_train)
error_cuadratico = mean_squared_error(y_train, y_pred)
mostrar_variables(0, 0, score, error_cuadratico)


# %% [markdown]
# #### Predecir por las características del atleta


# %%
y_pred = dtc.predict(np.array([[edad, peso, altura]]))
prediccion = y_pred[0] 
print('La probabilidad de ganar medalla de oro es : {:.3f}'.format(prediccion))
resultados_oro.append(['Arvol de desición', score, 0, 0, error_cuadratico, prediccion])


# %% [markdown]
# ### Probabilidad de ganar medalla de **plata**


# %%
X_train, X_test, y_train, y_test = generar_entrenamiento(data_plata, X_columnas, y_columnas)
dtc = DecisionTreeClassifier()
dtc.fit(X_train, y_train)
y_pred = dtc.predict(X_train)
score = dtc.score(X_train, y_train)
error_cuadratico = mean_squared_error(y_train, y_pred)
mostrar_variables(0, 0, score, error_cuadratico)


# %% [markdown]
# #### Predecir por las características del atleta


# %%
y_pred = dtc.predict(np.array([[edad, peso, altura]]))
prediccion = y_pred[0] 
print('La probabilidad de ganar medalla de plata es : {:.3f}'.format(prediccion))
resultados_plata.append(['Arvol de desición', score, 0, 0, error_cuadratico, prediccion])


# %% [markdown]
# ### Probabilidad de ganar medalla de **bronce**


# %%
X_train, X_test, y_train, y_test = generar_entrenamiento(data_bronce, X_columnas, y_columnas)
dtc = DecisionTreeClassifier()
dtc.fit(X_train, y_train)
y_pred = dtc.predict(X_train)
score = dtc.score(X_train, y_train)
error_cuadratico = mean_squared_error(y_train, y_pred)
mostrar_variables(0, 0, score, error_cuadratico)


# %% [markdown]
# #### Predecir por las características del atleta


# %%
y_pred = dtc.predict(np.array([[edad, peso, altura]]))
prediccion = y_pred[0] 
print('La probabilidad de ganar medalla de bronce es : {:.3f}'.format(prediccion))
resultados_bronce.append(['Arvol de desición', score, 0, 0, error_cuadratico, prediccion])


# %% [markdown]
# ## KNeighborsClassifier


# %% [markdown]
# ### Probabilidad general de obtener medallas# %% [markdown]


# %%
from sklearn.neighbors import KNeighborsClassifier

# %%
X_columnas = ['Edad', 'Peso', 'Altura']
y_columnas = 'MedallaNum'
X_train, X_test, y_train, y_test = generar_entrenamiento(data_new, X_columnas, y_columnas)
knc = KNeighborsClassifier()
knc.fit(X_train, y_train)
y_pred = knc.predict(X_train)
score = knc.score(X_train, y_train)
error_cuadratico = mean_squared_error(y_train, y_pred)
mostrar_variables(0, 0, score, error_cuadratico)

# %% [markdown]
# #### Predecir por las características del atleta


# %%
y_pred = knc.predict(np.array([[edad, peso, altura]]))
prediccion = y_pred[0] 
print('La probabilidad de ganar medalla es : {:.3f}'.format(prediccion))
resultados.append(['KNeighbors classifier', score, 0, 0, error_cuadratico, prediccion])


# %% [markdown]
# ### Probabilidad de ganar medalla de **oro**


# %%
X_train, X_test, y_train, y_test = generar_entrenamiento(data_oro, X_columnas, y_columnas)
knc = KNeighborsClassifier()
knc.fit(X_train, y_train)
y_pred = knc.predict(X_train)
score = knc.score(X_train, y_train)
error_cuadratico = mean_squared_error(y_train, y_pred)
mostrar_variables(0, 0, score, error_cuadratico)

# %% [markdown]
# #### Predecir por las características del atleta


# %%
y_pred = knc.predict(np.array([[edad, peso, altura]]))
prediccion = y_pred[0] 
print('La probabilidad de ganar medalla de oro es : {:.3f}'.format(prediccion))
resultados_oro.append(['KNeighbors classifier', score, 0, 0, error_cuadratico, prediccion])


# %% [markdown]
# ### Probabilidad de ganar medalla de **plata**


# %%
X_train, X_test, y_train, y_test = generar_entrenamiento(data_plata, X_columnas, y_columnas)
knc = KNeighborsClassifier()
knc.fit(X_train, y_train)
y_pred = knc.predict(X_train)
score = knc.score(X_train, y_train)
error_cuadratico = mean_squared_error(y_train, y_pred)
mostrar_variables(0, 0, score, error_cuadratico)

# %% [markdown]
# #### Predecir por las características del atleta


# %%
y_pred = knc.predict(np.array([[edad, peso, altura]]))
prediccion = y_pred[0] 
print('La probabilidad de ganar medalla de plata es : {:.3f}'.format(prediccion))
resultados_plata.append(['KNeighbors classifier', score, 0, 0, error_cuadratico, prediccion])


# %% [markdown]
# ### Probabilidad de ganar medalla de **bronce**


# %%
X_train, X_test, y_train, y_test = generar_entrenamiento(data_bronce, X_columnas, y_columnas)
knc = KNeighborsClassifier()
knc.fit(X_train, y_train)
y_pred = knc.predict(X_train)
score = knc.score(X_train, y_train)
error_cuadratico = mean_squared_error(y_train, y_pred)
mostrar_variables(0, 0, score, error_cuadratico)

# %% [markdown]
# #### Predecir por las características del atleta


# %%
y_pred = knc.predict(np.array([[edad, peso, altura]]))
prediccion = y_pred[0] 
print('La probabilidad de ganar medalla de bronce es : {:.3f}'.format(prediccion))
resultados_bronce.append(['KNeighbors classifier', score, 0, 0, error_cuadratico, prediccion])


# %% [markdown]
# ## naive bayes


# %% [markdown]
# ### Probabilidad general de obtener medallas# %% [markdown]


# %%
from sklearn.naive_bayes import GaussianNB


# %%
X_columnas = ['Edad', 'Peso', 'Altura']
y_columnas = 'MedallaNum'
X_train, X_test, y_train, y_test = generar_entrenamiento(data_new, X_columnas, y_columnas)
nv = GaussianNB()
nv.fit(X_train, y_train)
y_pred = nv.predict(X_train)
score = nv.score(X_train, y_train)
error_cuadratico = mean_squared_error(y_train, y_pred)
mostrar_variables(0, 0, score, error_cuadratico)

# %% [markdown]
# #### Predecir por las características del atleta


# %%
y_pred = nv.predict(np.array([[edad, peso, altura]]))
prediccion = y_pred[0] 
print('La probabilidad de ganar medalla es : {:.3f}'.format(prediccion))
resultados.append(['naive bayes', score, 0, 0, error_cuadratico, prediccion])


# %% [markdown]
# ### Probabilidad de ganar medalla de **oro**


# %%
X_train, X_test, y_train, y_test = generar_entrenamiento(data_oro, X_columnas, y_columnas)
nv = GaussianNB()
nv.fit(X_train, y_train)
y_pred = nv.predict(X_train)
score = nv.score(X_train, y_train)
error_cuadratico = mean_squared_error(y_train, y_pred)
mostrar_variables(0, 0, score, error_cuadratico)

# %% [markdown]
# #### Predecir por las características del atleta


# %%
y_pred = nv.predict(np.array([[edad, peso, altura]]))
prediccion = y_pred[0] 
print('La probabilidad de ganar medalla de oro es : {:.3f}'.format(prediccion))
resultados_oro.append(['naive bayes', score, 0, 0, error_cuadratico, prediccion])



# %% [markdown]
# ### Probabilidad de ganar medalla de **plata**


# %%
X_train, X_test, y_train, y_test = generar_entrenamiento(data_plata, X_columnas, y_columnas)
nv = GaussianNB()
nv.fit(X_train, y_train)
y_pred = nv.predict(X_train)
score = nv.score(X_train, y_train)
error_cuadratico = mean_squared_error(y_train, y_pred)
mostrar_variables(0, 0, score, error_cuadratico)

# %% [markdown]
# #### Predecir por las características del atleta


# %%
y_pred = nv.predict(np.array([[edad, peso, altura]]))
prediccion = y_pred[0] 
print('La probabilidad de ganar medalla de plata es : {:.3f}'.format(prediccion))
resultados_plata.append(['naive bayes', score, 0, 0, error_cuadratico, prediccion])



# %% [markdown]
# ### Probabilidad de ganar medalla de **bronce**


# %%
X_train, X_test, y_train, y_test = generar_entrenamiento(data_bronce, X_columnas, y_columnas)
nv = GaussianNB()
nv.fit(X_train, y_train)
y_pred = nv.predict(X_train)
score = nv.score(X_train, y_train)
error_cuadratico = mean_squared_error(y_train, y_pred)
mostrar_variables(0, 0, score, error_cuadratico)

# %% [markdown]
# #### Predecir por las características del atleta


# %%
y_pred = nv.predict(np.array([[edad, peso, altura]]))
prediccion = y_pred[0] 
print('La probabilidad de ganar medalla de bronce es : {:.3f}'.format(prediccion))
resultados_bronce.append(['naive bayes', score, 0, 0, error_cuadratico, prediccion])


# %% [markdown]
# ## Random forest regressor


# %% [markdown]
# ### Probabilidad general de obtener medallas# %% [markdown]


# %%
from sklearn.ensemble import RandomForestRegressor


# %%
X_columnas = ['Edad', 'Peso', 'Altura']
y_columnas = 'MedallaNum'
X_train, X_test, y_train, y_test = generar_entrenamiento(data_new, X_columnas, y_columnas)
rfr = RandomForestRegressor()
rfr.fit(X_train, y_train)
y_pred = rfr.predict(X_train)
score = rfr.score(X_train, y_train)
error_cuadratico = mean_squared_error(y_train, y_pred)
mostrar_variables(0, 0, score, error_cuadratico)

# %% [markdown]
# #### Predecir por las características del atleta


# %%
y_pred = rfr.predict(np.array([[edad, peso, altura]]))
prediccion = y_pred[0] 
print('La probabilidad de ganar medalla es : {:.3f}'.format(prediccion))
resultados.append(['Random forest regressor', score, 0, 0, error_cuadratico, prediccion])


# %% [markdown]
# ### Probabilidad de ganar medalla de **oro**


# %%
X_train, X_test, y_train, y_test = generar_entrenamiento(data_oro, X_columnas, y_columnas)
rfr = RandomForestRegressor()
rfr.fit(X_train, y_train)
y_pred = rfr.predict(X_train)
score = rfr.score(X_train, y_train)
error_cuadratico = mean_squared_error(y_train, y_pred)
mostrar_variables(0, 0, score, error_cuadratico)

# %% [markdown]
# #### Predecir por las características del atleta


# %%
y_pred = rfr.predict(np.array([[edad, peso, altura]]))
prediccion = y_pred[0] 
print('La probabilidad de ganar medalla de oro es : {:.3f}'.format(prediccion))
resultados_oro.append(['Random forest regressor', score, 0, 0, error_cuadratico, prediccion])



# %% [markdown]
# ### Probabilidad de ganar medalla de **plata**


# %%
X_train, X_test, y_train, y_test = generar_entrenamiento(data_plata, X_columnas, y_columnas)
rfr = RandomForestRegressor()
rfr.fit(X_train, y_train)
y_pred = rfr.predict(X_train)
score = rfr.score(X_train, y_train)
error_cuadratico = mean_squared_error(y_train, y_pred)
mostrar_variables(0, 0, score, error_cuadratico)

# %% [markdown]
# #### Predecir por las características del atleta


# %%
y_pred = rfr.predict(np.array([[edad, peso, altura]]))
prediccion = y_pred[0] 
print('La probabilidad de ganar medalla de plata es : {:.3f}'.format(prediccion))
resultados_plata.append(['Random forest regressor', score, 0, 0, error_cuadratico, prediccion])


# %% [markdown]
# ### Probabilidad de ganar medalla de **bronce**


# %%
X_train, X_test, y_train, y_test = generar_entrenamiento(data_bronce, X_columnas, y_columnas)
rfr = RandomForestRegressor()
rfr.fit(X_train, y_train)
y_pred = rfr.predict(X_train)
score = rfr.score(X_train, y_train)
error_cuadratico = mean_squared_error(y_train, y_pred)
mostrar_variables(0, 0, score, error_cuadratico)

# %% [markdown]
# #### Predecir por las características del atleta


# %%
y_pred = rfr.predict(np.array([[edad, peso, altura]]))
prediccion = y_pred[0] 
print('La probabilidad de ganar medalla de bronce es : {:.3f}'.format(prediccion))
resultados_bronce.append(['Random forest regressor', score, 0, 0, error_cuadratico, prediccion])


# %% [markdown]
# Compilando resultados


# %%
df_resultados = pd.DataFrame(resultados, columns=['Modelo', 'Score', 'Interseccion', 'Coeficiente', 'Error cuadratico', 'Prediccion'])
df_resultados = df_resultados.sort_values(['Prediccion', 'Score'], ascending=False).reset_index()
df_resultados_oro = pd.DataFrame(resultados_oro, columns=['Modelo', 'Score', 'Interseccion', 'Coeficiente', 'Error cuadratico', 'Prediccion'])
df_resultados_oro = df_resultados_oro.sort_values(['Prediccion', 'Score'], ascending=False).reset_index()
df_resultados_plata = pd.DataFrame(resultados_plata, columns=['Modelo', 'Score', 'Interseccion', 'Coeficiente', 'Error cuadratico', 'Prediccion'])
df_resultados_plata = df_resultados_plata.sort_values(['Prediccion', 'Score'], ascending=False).reset_index()
df_resultados_bronce = pd.DataFrame(resultados_bronce, columns=['Modelo', 'Score', 'Interseccion', 'Coeficiente', 'Error cuadratico', 'Prediccion'])
df_resultados_bronce = df_resultados_bronce.sort_values(['Prediccion', 'Score'], ascending=False).reset_index()


# %% [markdown]
# # Resumen de resultados
# Han sido ejecutados todos los modelos requeridos, ahora será visualizado cual de los modelos fue el 
# que mejor predijo la obtención de medallas para las 
# características del atleta ingresadas.

# %% [markdown]
# ## Datos de atleta

# %%
print('Edad    :', edad, sep='\t')
print('Peso     : ', peso, sep='\t')
print('Altura   : ', altura, sep='\t')
print('Sexo     : ', 'Femenino' if sexo == 0 else 'Masculino', sep='\t')
print('Deporte  :', deporte, sep='\t')

# %%
print("El mejor resultado general fue")
df_resultados.loc[0:0, ['Modelo', 'Prediccion', 'Score']]

# %%
print("El mejor resultado para medallas de oro fue")
df_resultados_oro.loc[0:0, ['Modelo', 'Prediccion', 'Score']]

# %%
print("El mejor resultado para medallas de plata fue")
df_resultados_plata.loc[0:0, ['Modelo', 'Prediccion', 'Score']]

# %%
print("El mejor resultado para medallas de bronce fue")
df_resultados_bronce.loc[0:0, ['Modelo', 'Prediccion', 'Score']]

# %% [markdown]
# ## Resultados completos de ejecución de los modelos

# %%
print("Tabla de resultados general")
df_resultados


# %%
print("Tabla de resultados medallas de oro")
df_resultados_oro


# %%
print("Tabla de resultados medallas de plata")
df_resultados_plata


# %%
print("Tabla de resultados medallas de bronce")
df_resultados_bronce


# %% [markdown]
# # Conclusión
# En primer lugar fue necesario realizar una exploracion y analisis al dataset de  
# juegos olímpicos, para posterior mente procesar los datos para una 
# óptima ejecución de los modelos.<br>
# Fue necesario procesar los datos para evitar que los modelos sesgarán la información por los valores mayores 
# y evitar que memorizarán los datos introducidos, para eso se realizo distintos procedimientos que permitieron ajustar los datos de mejor
# manera sin perjuicio de los resultados obtenidos.<br>
# Finalmente, se aprecia de como las características de un atleta
# puede ser procesado para poder predecir que comportamiento 
# poseerá en los próximos juegos olímpicos, cabe señalar que, los datos utilizados (como pruebas)
# corresponden a la atleta Kristel Koubrick, pero de igual manera pueden ser 
# cambiados para obtener resultados con otro participante.<br>
# Las ejecuciones de los modelos cambian al correr estos nuevamente, pero los resultados se 
# adaptan para absorber ese cambio y mostrar la nueva ejecución, permitiendo 
# visualizar la participación de mas atletas e inclusive re-evaluar al mismo.


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


