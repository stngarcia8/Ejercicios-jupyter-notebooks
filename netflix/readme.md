##Ejercicio Netflix

La estructura de del ejercicio es la siguiente:
Netflix
    data
        final
        intermediate
        original
    notebook
    script

data: carpeta que contiene los archivos de datos necesarios para la ejecucion de los notebooks, posee los siguientes directorios.
- original 
    netflix.csv (datos originales del ejercicio)
- intermediate 
    netflix-actores.csv (listado de actores extraidos del archivo original procesado )
    netflix-categorizado.pkl (archivo de datos categorizado para las consultas requeridas)
    netflix-clasificacion.csv (archivo que posee las clasificaciones extraidas del archivo de datos original procesado)
    netflix-directores.csv (archivo de directores extraidos desde el archivo original procesado)
    netflix-limpio.csv (archivo de datos procesado que tiene los valores limpios en comparacion al archivo original)
    netflix-paises.csv (paises extraidos desde el archivo original procesado)
- final (archivos de datos resultantes del ejercicio)

notebook: Contiene los archivos de tipo notebook de jupyter, los archivos que lo componen son
-   p01-explorar-datos.ipynb (notebook de exploracion inicial de los datos)
- p02-transformar-datos.ipynb (notebook para la transformacion de los datos)
- p03-desarrollar-datos.ipynb (notebook con los resultados que han sido solicitados)

report: carpeta que tiene los notebook exportados a archivos html, contiene los siguientes archivos:
- p01-explorar-datos.html
- p02-transformar-datos.html
- p03-desarrollar-datos.html

script: Contiene los script con los que fue realizado el ejercicio, posee los archivos:
- p01-exploracion.py (script para generar la exploracion inicial de los datos)
- p02-transformacion.py (script para generar la transformacion de los datos iniciales)
- p03-desarrollo.py (desarrollo de las preguntas solicitadas por el ejercicio)



