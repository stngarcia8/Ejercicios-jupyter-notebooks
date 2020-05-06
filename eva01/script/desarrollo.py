# %% [markdown]
# # Evaluación actividad Juegos Olímpicos
# Alumno: Daniel García Loyola<br>
# Profesor: Israel Naranjo Retamal<br>
# Ramo: Minería de datos (MDY7101)<br>
# Sección: 002D<br>
# Entrega: 06/05/2020

# %% 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns

# %% [markdown]
# ## Cargando datos de juegos olímpicos

# %%
data_file="../data/original/JuegosOlimpicos.csv"
data = pd.read_csv(data_file)
data.shape 


# %% [markdown]
# Ajustando títulos de columnas al dataframe
# %%
data.columns=["ID", "NombreAtleta", "Sexo", "Edad", "Altura", "Peso", "Equipo", "Comite", "JuegosOlimpicos", "Año", "Temporada", "Ciudad", "Deporte", "Competencia", "Medalla"]
data.columns

# %%
data.head()



# %% [markdown]
# ## Explorando datos de juegos olímpicos

# %%
data.describe(include="all")

# %% [markdown]
# ## Visualizando info del dataframe

# %%
data.info()


# %% [markdown]
# ## Verificando tipos de datos

# %%
data.dtypes


# %% [markdown]
# ## Visualizando elementos nulos

# %%
data["Medalla"].isnull().sum()



# %% [markdown]
# # Anecdotas de los juegos olímpicos


# %% [markdown]
# <table align="left" width="75%">
# <td width="20%" align="center"><img src="../imagenes/pregunta01.png" width="100px" heigth="75px"/></td>
# <td width="75%" style="text-align: justify;">
# 1. Dorando Pietri el perdedor más famoso de la historia. El atleta italiano, pastelero de profesión, se desplomó varias veces antes de llegar a la meta. Un juez y el entonces periodista Arthur Conan Doyle le arrastraron hasta la línea de llegada. ¿En qué año sucedió esta anécdota?
# </td>
# </table>

# %%
atleta = data.loc[:, "NombreAtleta"]=="Dorando Pietri"
competencia = data.loc[:, "Competencia"].str.contains("Marathon")
anio = data.loc[:, "Año"]==1908
dorando = data.loc[atleta & competencia & anio]
dorando[["Año", "Temporada", "Ciudad", "Deporte", "Competencia"]]

# %% [markdown]
# <p style="text-align: justify; width: 75%;">
# Dorando fue aclamado por su fortaleza al competir y perder multiples veces la conciencia durante la maratón de las olimpiadas de Londres en 1908, pero aunque fue descalificado, igualmente llego al final de la competencia.
# </p><hr>



# %% [markdown]
# <table align="left" width="75%">
# <td width="75%" style="text-align: justify;">
# 2. Wilma Rudolph junto a su elegancia y belleza, le valió el sobrenombre de Gacela negra es una de esas historias de superación que emociona. De familia humilde y vigésima de veintidós hermanos, pasó varios años sin caminar cuando era pequeña a causa de una poliomielitis. ¿Cuál(es) son sus logros que la llevaron a tener el reconocimiento de superación?
# </td>
# <td width="20%" align="center"><img src="../imagenes/pregunta02.png" width="100px" heigth="75px"/></td>
# </table>

# %%
nombre = data.loc[:, "NombreAtleta"].str.contains("Wilma")
apellido = data.loc[:, "NombreAtleta"].str.contains("Rudolph")
wilma = data.loc[nombre & apellido]
wilma[["Edad", "Año", "Temporada", "Ciudad", "Deporte", "Competencia", "Medalla"]]

# %% [markdown]
# <p style="text-align: justify; width: 75%;">
# Wilma Rudolph, participo por primera vez en las olimpiadas de Melbourne en el año 1956 con tan solo 16 años, participó en dos disciplinas 200 metros planos y en el relevo 4x100 metros, en esta última competición obtubo su primera medalla de bronce.
# En los juegos olímpicos de Roma el año 1960, obtubo tres medallas de oro con tan solo 20 años de edad, las competiciones que gano fueron 100 y 200 metros planos y en el relevo de 4x100 metros.
# </p><hr>



# %% [markdown]
# <table align="left" width="75%">
# <td width="20%" align="center"><img src="../imagenes/pregunta03.png" width="100px" heigth="75px"/></td>
# <td width="75%" style="text-align: justify;">
# 3. La esgrimista Helene Mayer fue la única judía que desfiló bajo la bandera Nazi y estrechó la mano de Hitler al recibir su medalla de plata. ¿En qué JJOO sucedió?
# </td>
# </table>

# %%
nombre = data.loc[:, "NombreAtleta"].str.contains("Helene")
apellido = data.loc[:, "NombreAtleta"].str.contains("Mayer")
anio = data.loc[:, "Año"]==1936
helene = data.loc[nombre & apellido & anio]
helene[["Ciudad", "Año", "Deporte", "Competencia", "Medalla"]]

# %% [markdown]
# <p style="text-align: justify; width: 75%;">
# Helene Mayer en el año 1936, en los juegos realizados en Berlin Alemania, gano la medalla de plata en esgrima femenina representando a Alemania, esto permitio que los juegos se realizaran y finalmente evito el boicot internacional, ya que Hittler no quería que ningún deportista Judío representara a la Alemania Nazi de ese entonces.
# </p><hr>



# %% [markdown]
# <table align="left" width="75%">
# <td width="75%" style="text-align: justify;">
# 4. Bob Beamon el monstruo que saltó ocho metros noventa centímetros, en el salto del siglo ¿En qué temporada de los JJOO lo realizo?
# </td>
# <td width="20%" align="center"><img src="../imagenes/pregunta04.png" width="100px" heigth="75px"/></td>
# </table>

# %%
nombre = data.loc[:, "NombreAtleta"].str.contains("Bob")
apellido = data.loc[:, "NombreAtleta"].str.contains("Beamon")
anio = data.loc[:, "Año"]==1968
bob = data.loc[nombre & apellido & anio]
bob[["Ciudad", "Año", "JuegosOlimpicos",  "Deporte", "Competencia", "Medalla"]]

# %% [markdown]
# <p style="text-align: justify; width: 75%;">
# Robert Bob Beamon obtuvo el record de salto de longitud  alcanzando la distancia de 8 metros y 90 centimetros en la ciudad de Mejico en el año 1968, en los juegos olímpicos de verano, dicho récord perduró durante 22 años, 10 meses y 22 días, hasta ser superado en la final del Mundial de Tokio 1991 por Mike Powell (8,95 metros).
# </p><hr>



# %% [markdown]
# <table align="left" width="75%">
# <td width="20%" align="center"><img src="../imagenes/pregunta05.png" width="100px" heigth="75px"/></td>
# <td width="75%" style="text-align: justify;">
# 5. Jim Thorpe indio piel roja, ganador olímpico, vio truncada su carrera cuando le retiraron las medallas y no fue hasta 70 años después cuando Samaranch reconoció el error y devolvió las insignias a sus hijos. ¿Dónde logro las medallas y en que disciplina?
# </td>
# </table>

# %%
nombre = data.loc[:, "NombreAtleta"].str.contains("Jim")
apellido = data.loc[:, "NombreAtleta"].str.contains("Thorpe")
medallas = data.loc[:, "Medalla"]=="Gold"
jim = data.loc[nombre & apellido & medallas]
jim[["JuegosOlimpicos", "Ciudad", "Deporte", "Competencia", "Medalla"]]

# %% [markdown]
# <p style="text-align: justify; width: 75%;">
#  Jim Thorpe gano las medallas de oro en el Pentatlon y Decatlon en las olimpiadas de verano en 1912, realizadas en Estocolmo, estas fueron retiradas ya que Jim habia jugado béisbol de manera profesional antes de presentarse a las olimpiadas.
# </p><hr>



# %% [markdown]
# <table align="left" width="75%">
# <td width="75%" style="text-align: justify;">
# 6. ¿Puede alguien ganar una maratón corriendo descalzo y tres semanas después de ser operado de apendicitis? El etíope Abebe Bikila lo logró más tarde quedó parapléjico, pero no se rindió y siguió compitiendo con el mismo espíritu de lucha. ¿Sabes en que año logro su hazaña?
# </td>
# <td width="20%" align="center"><img src="../imagenes/pregunta06.png" width="100px" heigth="75px"/></td>
# </table>

# %%
nombre = data.loc[:, "NombreAtleta"].str.contains("Abebe")
apellido = data.loc[:, "NombreAtleta"].str.contains("Bikila")
anio = data.loc[:, "Año"]==1960
abebe = data.loc[nombre & apellido & anio]
abebe[["JuegosOlimpicos", "Ciudad", "Deporte", "Competencia", "Medalla"]]

# %% [markdown]
# <p style="text-align: justify; width: 75%;">
# Lo consiguio en los juegos olímpicos de Roma en 1960, además estableció una nueva marca en la maratón teniendo un tiempo de 2 horas con doce minutos y doce segundos, en 1969 el atleta se vio involucrado en un accidente de autos cerca de Adís Abeba, en Etiopía, lo que le produjo una paraplejia. Bikila nunca pudo reponerse totalmente del accidente, y falleció a los 41 años de edad.
# </p><hr>



# %% [markdown]
# <table align="left" width="75%">
# <td width="20%" align="center"><img src="../imagenes/pregunta07.png" width="100px" heigth="75px"/></td>
# <td width="75%" style="text-align: justify;">
# 7. Abandonado al nacer, a los nueve años fumaba y bebía. Vivió con miedos y fobias, condicionado por su elección sexual. Dura realidad la de Louganis que acabó transformando en éxito y superación, al obtener varios oros olímpicos. ¿Sabes en que disciplina logro el podio?
# </td>
# </table>

# %%
apellido = data.loc[:, "NombreAtleta"].str.contains("Louganis")
medallas = data.loc[:, "Medalla"].isin(["Gold", "Silver", "Bronze"])
greg = data.loc[apellido & medallas]
greg[["NombreAtleta", "Edad", "Deporte", "Competencia", "JuegosOlimpicos", "Ciudad", "Medalla"]]

# %% [markdown]
# <p style="text-align: justify; width: 75%;">
# Greg Louganis, compitio en cuatro olimpiadas, consiguió medalla de plata en los juegos de 1976 en Montreal (salto en plataforma) con tan solo 16 años de edad, luego participó en juegos realizados en Los Angeless 1984 y Seoul 1988 logrando medallas de oro en las especialidades de salto en trampolín y salto en plataformas consecutivamente.
# </p><hr>




# %% [markdown]
# # Anecdotas de Chilenos en juegos olímpicos

# %% [markdown]
# Definiendo variables comunes para estos deportistas.

# %%
equipo = data.loc[:, "Equipo"]=="Chile"
campos =["NombreAtleta", "JuegosOlimpicos", "Ciudad", "Deporte", "Competencia"]


# %% [markdown]
# <table align="left" width="75%">
# <td width="75%" style="text-align: justify;">
# 8. Martín Vargas pelea con pañal<br>A los 17 años, Martín Vargas asistió a los JJ.OO, pero solo alcanzó a combatir una vez, ya que fue eliminado por el colombiano Calixto Pérez en estrecho veredicto. Su explicación fue contundente: "Estaba enfermo del estómago por comer fruta mal lavada. Tuve que pelear con pañales, lo que me perjudicó. En aquella ocasión Pérez me calzó un derechazo a la cara que me tiró al suelo. Me levanté prontamente y no hubo cuenta, pero estaba nocaut. Todo el estadio nos aplaudió mucho”. 
# </td>
# <td width="20%" align="center"><img src="../imagenes/pregunta08.png" width="100px" heigth="75px"/></td>
# </table>

# %%
apellido = data.loc[:, "NombreAtleta"].str.contains("Vargas Fuentes")
dale_martin = data.loc[apellido & equipo]
dale_martin[["NombreAtleta", "Competencia", "JuegosOlimpicos", "Ciudad"]]

# %% [markdown]
# <p style="text-align: justify; width: 75%;">
# Aquella pelea de Martin Vargas ocurrio en los juegos olímpicos de Munich en 1972, en la categoría de peso mosca.
# </p><hr>



# %% [markdown]
# <table align="left" width="75%">
# <td width="20%" align="center"><img src="../imagenes/pregunta09.png" width="100px" heigth="75px"/></td>
# <td width="75%" style="text-align: justify;">
# 9. El primer eliminado por indisciplina<br>Francisco Arellano, el "Mono", era hermano de David, el mártir colocolino fallecido en 1927, en Valladolid. Un año después de la desgracia, era uno de los pilares del equipo de fútbol que perdió con Portugal 4 a 2 en la primera fase. Arellano no jugó, pues fue devuelto en el mismo barco en el que había llegado. "Me echaron sin escucharme. No fue mi culpa, lo hice para defender a un compañero. Me devolvieron con poca plata y en Buenos Aires pasé un día sin comer". La prensa de la época no da detalles del incidente (aparentemente, una pelea a golpes), aunque otro hermano, Alberto, fundador del Círculo de Periodistas Deportivos, lo defendió: "Volvió para asistir al funeral de nuestra madre".
# </td>
# </table>

# %%
deporte = data.loc[:, "Deporte"]=="Football"
anio = data.loc[:, "Año"]==1928
arellano = data.loc[deporte & anio & equipo]
arellano[["NombreAtleta", "Equipo", "Deporte", "JuegosOlimpicos", "Ciudad"]].head(22)

# %% [markdown]
# <p style="text-align: justify; width: 75%;">
# Este episodio ocurre en los juegos olímpicos de Amsterdam de 1928, la lista anterior se pueden visualizar los compañeros de Francisco Arellano que si participaron del certamen.
# </p><hr>



# %% [markdown]
# <table align="left" width="75%">
# <td width="95%" style="text-align: justify;">
# 10. La primera mujer a pan y té<br>Raquel Martínez recién se había titulado como profesora de Educación Física cuando el rector de la Universidad de Chile, Juvenal Hernández, encabezó la campaña para costear su pasaje, donde se convertiría en la primera mujer chilena en participar de los JJ.OO. Lo suyo rozó la tragedia. En el barco, según contó, "la comida era tan mala que casi todos se enfermaron. Yo opté por consumir solo té y pan durante 10 días. La fiebre me atacó y estuve en cama hasta que llegué a Frankfurt". Tras la odisea compitió. Fue última en su serie de 100 metros planos.
# </td>
# </table>

# %%
nombre = data.loc[:, "NombreAtleta"].str.contains("Raquel")
raquel = data.loc[nombre & equipo]
raquel[campos]

# %% [markdown]
# <p style="text-align: justify; width: 75%;">
# Raquel Martínez vivió esta lamentable situación en los juegos olímpicos de Berlin en el año 1936, compitiendo en los 100 metros planos en la división femenina.
# </p><hr>



# %% [markdown]
# <table align="left" width="75%">
# <td width="95%" style="text-align: justify;">
# 11. El blanco más rápido en el living<br>Iván Moreno llegó en su mejor momento a los JJ.OO Había adquirido experiencia en Tokio 64 y todo hacía pensar que elevaría sus marcas. El problema es que estuvo en cama dos días antes de su prueba, por una gripe. ¿La razón? Chile fue asignado al edificio 6, en el cuarto piso. Era un departamento que tenía tres dormitorios y un living, que fueron ocupados por orden de llegada. Y los primeros en llegar fueron los equitadores y boxeadores. Luego los tiradores y al final, los atletas, que debieron ocupar el living. "El lugar se transformó en pasillo y las corrientes de aire terminaron por resfriar a todo el equipo", confesó Jorge Grosser, otro atleta que tuvo fiebre. Moreno, en todo caso, igual respondió llegando a semifinales en 100 y 200 metros planos.
# </td>
# </table>

# %%
nombre = data.loc[:, "NombreAtleta"].str.contains("Iv")
apellido = data.loc[:, "NombreAtleta"].str.contains("Moreno")
anio = data.loc[:, "Año"]!=1964
ivan = data.loc[nombre & apellido & equipo & anio]
ivan[campos]

# %% [markdown]
# <p style="text-align: justify; width: 75%;">
# Esta anecdota ocurrió en los juegos olímpicos del año 1968 en Ciudad de Méjico.
# </p><hr>



# %% [markdown]
# <table align="left" width="75%">
# <td width="95%" style="text-align: justify;">
# 12. El primer chileno, ¿a los 14?<br>Nunca se ha sabido con certeza, pero los griegos tienen su registro propio y dicen que acudió un chileno de nombre Luis Subercaseaux Errázuriz, quien se entusiasmó, se inscribió, le prestaron indumentaria e ingresó a la historia. Habría corrido los 100 y 400 metros, aunque no hay registro de ubicación ni marcas. En la inscripción quedó como representante chileno gracias al pasaporte diplomático. Al cumplirse el centenario de su gesta, Ramón Subercaseaux, su hijo, agregó datos a la historia en "El Mercurio", señalando que mandó confeccionar una bandera chilena para lucirla en el evento. Si así hubiese ocurrido, Subercaseaux habría participado con 14 años, ya que nació en 1882.
# </td>
# </table>

# %%
apellido = data.loc[:, "NombreAtleta"].str.contains("Subercaseaux")
anio = data.loc[:, "Año"]==1896
luis = data.loc[apellido & anio]
luis[campos]

# %% [markdown]
# <p style="text-align: justify; width: 75%;">
# No es posible encontrar datos asociados a Luis Subercaseaux, se ha buscado la información por el apellido "Subercaseaux" y el año "1896" y los filtros no arrojan resultados.
# </p><hr>



# %% [markdown]
# <table align="left" width="75%">
# <td width="95%" style="text-align: justify;">
# 13. La crisis nerviosa de la hermana de Mund<br>Gunther Mund fue pionero en los saltos ornamentales en Chile. Participó en Londres con apenas 14 años, y en Melbourne ya tenía la experiencia y talento suficientes para estar en la final de la plataforma de tres metros, donde logra un meritorio séptimo lugar. A esos juegos viaja acompañado de su hermana Lilo, quien debía hospedarse junto a Marlene Ahrens en la Villa Olímpica, pero no pudo competir debido a que al ser separada de su hermano, "la atrapó el temor de la absoluta soledad que anulaba su personalidad y sufrió una crisis nerviosa", explicó su madre, Margarita.
# </td>
# </table>

# %%
nombre = data.loc[:, "NombreAtleta"].str.contains("Lilo")
apellido = data.loc[:, "NombreAtleta"].str.contains("Mund")
lilo = data.loc[nombre & apellido & equipo]
lilo[campos]

# %% [markdown]
# <p style="text-align: justify; width: 75%;">
# Lilo sufrio aquel incidente en los juegos olímpicos del año 1956 en Melbourne, estaba inscrita en la disciplina de salto de plataforma.
# </p><hr>



# %% [markdown]
# <table align="left" width="75%">
# <td width="95%" style="text-align: justify;">
# 14. ¿Cuál es el porcentaje de mujeres participantes en los Juegos Olímpicos desde 1896 a 2016?
# </td>
# </table>

# %%
filtro_femenino = data.loc[:, "Sexo"]=="F"
filtro_masculino = data.loc[:, "Sexo"]=="M"
mujeres_df = data.loc[filtro_femenino] 
hombres_df = data.loc[filtro_masculino] 
porcentaje_mujeres = round(((len(mujeres_df)/len(data))*100),2)
porcentaje_hombres = round(((len(hombres_df)/len(data))*100),2)
print("Total de participantes en juegos olímpicos: {}".format(len(data)))
print("Total participantes hombres: {} ({}%)".format(len(hombres_df), porcentaje_hombres))
print("Total participantes mujeres: {} ({}%)".format(len(mujeres_df), porcentaje_mujeres))

# %%
%matplotlib inline 
data.Sexo.groupby(data.Sexo).count().plot(kind="pie", legend="Reverse")
plt.title("Relación de participantes femeninas en juegos olímpicos")
plt.legend(["F - Femenino", "M - Masculino"])
plt.ylabel("")
plt.show()

# %% [markdown]
# <p style="text-align: justify; width: 75%;">
# De un total de 271116 participantes en los juegos olímpicos, desde los años 1986 hasta el 2016 han participado 74522 mujeres, esto representa un 27.49% de la muestra.<br>
# En la gráfica anterior se puede visualizar la relación entre participantes mujeres y participantes hombres, en el, se distingue la proporción entre ambos sexos.
# </p><hr>



# %% [markdown]
# <table align="left" width="75%">
# <td width="95%" style="text-align: justify;">
# 15. ¿En qué temporada de los Juegos Olímpicos compitieron por primera vez las atletas femeninas? 
# </td>
# </table>

# %%
temporada = mujeres_df.sort_values("Año", ascending=True)
temporada.iloc[0:1, 9:12]

# %% [markdown]
# <p style="text-align: justify; width: 75%;">
#  Los primeros juegos olímpicos que registra la participación de mujeres corresponde a los realizados en el verano del año 1900 en Francia.
# </p><hr>


# %% [markdown]
# <table align="left" width="75%">
# <td width="95%" style="text-align: justify;">
# 16. ¿Cuál es el porcentaje que representa la participación total de atletas femeninas en la primera temporada, respecto a los atletas masculinos?
# </td>
# </table>

# %%
anio = data.loc[:, "Año"]==1900
primera_temporada = data.loc[anio]
filtro_femenino = primera_temporada.loc[:, "Sexo"]=="F"
filtro_masculino = primera_temporada.loc[:, "Sexo"]=="M"
mujeres_df = primera_temporada.loc[filtro_femenino] 
hombres_df = primera_temporada.loc[filtro_masculino] 
porcentaje_mujeres = round(((len(mujeres_df)/len(primera_temporada))*100),2)
porcentaje_hombres = round(((len(hombres_df)/len(primera_temporada))*100),2)
print("Total de participantes en juegos olímpicos de 1900: {}".format(len(primera_temporada)))
print("Total participantes hombres: {} ({}%)".format(len(hombres_df), porcentaje_hombres))
print("Total participantes mujeres: {} ({}%)".format(len(mujeres_df), porcentaje_mujeres))

# %%
%matplotlib inline 
primera_temporada.Sexo.groupby(primera_temporada.Sexo).count().plot(kind="pie", legend="Reverse")
plt.title("Relación de participantes según genero en olimpiadas de 1900")
plt.legend(["F - Femenino", "M - Masculino"])
plt.ylabel("")
plt.show()

# %% [markdown]
# <p style="text-align: justify; width: 75%;">
# En la primera temporada de los juegos olímpicos del año 1896 en Atenas, no hubo participación femenina, por lo tanto para revisar la participación de mujeres en los juegos, se revisará los realizados en Francia en 1900, donde si estubieron participando atletas femeninas.<br>
# En Francia participaron 1936 atletas, de los cuales, 1903 fueron hombres representando un 98.3%, mientras tanto tan solo el 1.7% corresponde a atletas femeninas con una cantidad de 33 participantes.
# </p><hr>



# %% [markdown]
# <table align="left" width="75%">
# <td width="95%" style="text-align: justify;">
# 17. ¿Cuál es el porcentaje de las atletas femeninas en Río de Janeiro 2016, respecto al total de deportista?
# </td>
# </table>

# %%
anio = data.loc[:, "Año"]==2016
primera_temporada = data.loc[anio]
filtro_femenino = primera_temporada.loc[:, "Sexo"]=="F"
filtro_masculino = primera_temporada.loc[:, "Sexo"]=="M"
mujeres_df = primera_temporada.loc[filtro_femenino] 
hombres_df = primera_temporada.loc[filtro_masculino] 
porcentaje_mujeres = round(((len(mujeres_df)/len(primera_temporada))*100),2)
porcentaje_hombres = round(((len(hombres_df)/len(primera_temporada))*100),2)
print("Total de participantes en juegos olímpicos de 2016: {}".format(len(primera_temporada)))
print("Total participantes hombres: {} ({}%)".format(len(hombres_df), porcentaje_hombres))
print("Total participantes mujeres: {} ({}%)".format(len(mujeres_df), porcentaje_mujeres))

# %%
%matplotlib inline 
primera_temporada.Sexo.groupby(primera_temporada.Sexo).count().plot(kind="pie", legend="Reverse")
plt.title("Relación de participantes según genero en olimpiadas de Río de Janeiro en 2016")
plt.legend(["F - Femenino", "M - Masculino"])
plt.ylabel("")
plt.show()

# %% [markdown]
# <p style="text-align: justify; width: 75%;">
# En RRío de Janeiro participaron un total de 13688 atletas, la presencia femenina fue de un 45.46% con 6223 atletas, en cambio la participación de los hombres corresponde al 54.54% con 7465 competidores.
# </p><hr>



# %% [markdown]
# <table align="left" width="75%">
# <td width="95%" style="text-align: justify;">
# 18. ¿Sabes quién es el deportista que más medallas de oro ha ganado en los Juegos Olímpicos hasta el momento? Si quieres saber quién ha ganado más medallas en los Juegos Olímpicos, no dudes en aplicar un análisis que arroje los 10 deportistas con más medallas olímpicas de oro en sus palmareses.
# </td>
# </table>

# %%
filtro_medallas = data.loc[:, "Medalla"]=="Gold"
medallistas = data.loc[filtro_medallas]
print("Rankin de deportistas con mayor cantidad de medallas de oros ganadas.")
medallistas["NombreAtleta"].value_counts().head(10)

# %% [markdown]
# <p style="text-align: justify; width: 75%;">
# El top 3  de deportistas con mas medallas de oro ganadas, la lidera Michael Phelps con 23 medallas, lo sigue Raymond Clarence 10 medallas y por último Mark Andrew Spitz con 9 medallas de oro.<br>
# En el siguiente apartado se visualizarán donde obtubieron sus medallas estos deportistas.
# </p><hr>

# %%
medallas = data.loc[:, "Medalla"].isin(["Gold"])
campos =["NombreAtleta", "Equipo", "JuegosOlimpicos", "Ciudad", "Deporte", "Competencia", "Medalla"]

# %%
def mostrar_medallas(nombre_atleta, apellido_atleta):
    nombre = data.loc[:, "NombreAtleta"].str.contains(nombre_atleta)
    apellido = data.loc[:, "NombreAtleta"].str.contains(apellido_atleta)
    atleta  = data.loc[nombre & apellido & medallas]
    return atleta[campos]


# %% [markdown]
# ### Michael Phelps

# %%
mostrar_medallas("Michael", "Phelps").head(25)

# %% [markdown]
# ### Raymond Clarence

# %%
mostrar_medallas("Raymond", "Clarence").head(15)

# %% [markdown]
# ### Mark Andrew Spitz

# %%
mostrar_medallas("Mark","Spitz").head(10)



# %% [markdown]
# <table align="left" width="75%">
# <td width="95%" style="text-align: justify;">
# 19. ¿Cuál son los datos del primer atleta que se coronó campeón en los Juegos Olímpicos? Y en el apartado femenino ¿Quién fue? 
# </td>
# </table>

# %%
nombre = data.loc[:, "NombreAtleta"].str.contains("James")
apellido = data.loc[:, "NombreAtleta"].str.contains("Connolly")
anio = data.loc[:, "Año"]==1896
primer_medallista = data.loc[nombre & apellido & anio]
primer_medallista[campos]

# %% [markdown]
# <p style="text-align: justify; width: 75%;">
# El Estado Unidense James Connolly, fue el primer campeón olímpico al ganar medallas en las disciplinas de triple salto, salto alto y salto largo, esto lo logro el 06/04/1896.
# </p><hr>

# %%
nombre = data.loc[:, "NombreAtleta"].str.contains("Charlotte")
apellido = data.loc[:, "NombreAtleta"].str.contains("Cooper")
primera_medallista = data.loc[nombre & apellido]
primera_medallista[campos]

# %% [markdown]
# <p style="text-align: justify; width: 75%;">
# Charlotte Cooper fue la primera mujer en ganar un título olímpico, Ganó en la final a la francesa Hélène Prévost por 6-1, 7-5.<br>
# En las mismas olimpiadas sumó otra victoria en la disciplina de dobles mixto con su compañero Reginald Doherty 6-2, 6-4, sobre Harold Mahony y Prévost.
# </p><hr>



# %% [markdown]
# <table align="left" width="75%">
# <td width="95%" style="text-align: justify;">
# 20. ¿Cuáles son los 5 primeros países de Latinoamérica en cuanto a obtenciones de medallas de oro en todos los JJ.OO? 
# </td>
# </table>

# %%
lista_paises = ["Argentina", "Brazil", "Bolivia", "Chile", "Colombia", "Ecuador", "Paraguay", "Peru", "Uruguay", "Venezuela"]
paises_latinos  = data.loc[data["Equipo"].isin(lista_paises), ["Equipo", "Medalla"]]
paises_latinos["Equipo"].value_counts()

# %%
%matplotlib inline 
paises_latinos.groupby("Equipo")["Equipo"].count().plot(kind="barh", legend="Reverse")
plt.xlabel("Cantidad de medallas")
plt.ylabel("Paises")
plt.title("Ranking de medallas obtenidas por paises latinoamericanos")
plt.show()

# %% [markdown]
# <p style="text-align: justify; width: 75%;">
# Los paises con más medallas obtenidases liderado por Brasil con 3772, lo sigue Argentina con 3199 y en tercer lugar se encuentra Colombia con 1068 medallas.<br>
# A continuación se mostrará la relación existente entre tipos de medallas obtenidas por paises latinoamericanos.
# </p><hr>

# %%
print("Cantidad de tipo de medalla obtenidas por paises latinoamericanos:")
paises_latinos["Medalla"].value_counts()

# %%
%matplotlib inline 
paises_latinos.groupby("Medalla")["Medalla"].count().plot(kind="barh", legend="Reverse")
plt.title("Relación de tipos de medallas obtenidas en latinoamerica")
plt.xlabel("Cantidad de medallas")
plt.ylabel("Tipo de medalla")
plt.show()

# %% [markdown]
# En la gráfica podemos distinguir que la mayor cantidad de medallas obtenidas por los paises latino americanos son de tipo bronce (350), seguidas por medallas de plata (299) y por último medallas de oro (237).

# %% [markdown]
# <p style="text-align: justify; width: 75%;">
#  Por último se visualizarán los resultados separados por tipos de medallas y por paises.
# </p><hr>

# %%
fig=plt.subplots(figsize=(10,5))
d=sns.countplot(x='Equipo',data=paises_latinos,hue='Medalla',palette='bright')
m=plt.setp(d.get_xticklabels(),rotation=90)
plt.title("Relación de medallas por paises latinoamericanos")
plt.ylabel("Cantidad de medallas")
plt.xlabel("Paises")
plt.show()

# %% [markdown]
# <p style="text-align: justify; width: 75%;">
# Brasil posee la mayor cantidad de medallas en latinoamérica, la distribución indica que la gran cantidad de medallas obtenidas son de bronce seguidas por las medallas de plata y luego las de oro.<br>
# Sin embargo Argentina es mas pareja la obtención de medallas, estando casi eqilibradas las medallas de oro y bronce, seguidas muy de cerca las medallas de plata.<br>
# Por último los demás paises de la región poseen una cantidad muy inferior de medallas a excepción de Colombia, pero no alcanza a Argentina.
# </p><hr>

# %% [markdown]
# # Configuración utilizada para esta evaluación.

# %%
%load_ext watermark
%watermark