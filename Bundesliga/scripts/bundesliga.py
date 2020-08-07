# %%
import pandas as pd
import scipy.stats as ss


# %%
from IPython.display import IFrame
url = 'https://co.marca.com/claro/futbol-internacional/bundesliga/posiciones.html'
IFrame(url, width=970, height=600)


# %% [markdown]
# ## Cargando datos de Bundesliga

# %%
data_file = "../data/original/bundesliga.csv"
data = pd.read_csv(data_file, engine='python')
data.shape

# %%
tabla_equipos = data[['ID', 'Equipo']]
tabla_equipos.head(18)

# %%
tabla_posiciones = data[['ID', 'Equipo', 'PT:Puntos Totales', 'PJ:Partidos Jugados', 'PG:Partidos Ganados', 'PE:Partidos Empatados', 'PP:Partidos Perdidos', 'GF:Goles a favor', 'GC:Goles en contra']]
tabla_posiciones.columns = ['ID', 'Equipo', 'pt', 'pj', 'pg', 'pe', 'pp', 'gf', 'gc']
tabla_posiciones.head(18)

# %%
campaña_local = data[['ID', 'Equipo', 'PTL:Puntos Totales', 'PJL:Partidos Jugados', 'PGL:Partidos Ganados', 'PEL:Partidos Empatados', 'PPL:Partidos Perdidos', 'GFL:Goles a favor', 'GCL:Goles en contra']]
campaña_local.columns = ['ID', 'Equipo', 'pt', 'pj', 'pg', 'pe', 'pp', 'gf', 'gc']
campaña_local.head(18)

# %%
campaña_visita = data[['ID', 'Equipo', 'PTV:Puntos Totales', 'PJV:Partidos Jugados', 'PGV:Partidos Ganados', 'PEV:Partidos Empatados', 'PPV:Partidos Perdidos', 'GFV:Goles a favor', 'GCV:Goles en contra']]
campaña_visita.columns = ['ID', 'Equipo', 'pt', 'pj', 'pg', 'pe', 'pp', 'gf', 'gc']
campaña_visita.head(18)

# %%
media_goles_local = (campaña_local.gf.sum()/campaña_local.pj.sum())
media_goles_visita = (campaña_visita.gf.sum()/campaña_visita.pj.sum())

# %%
campaña_local['fal'] = campaña_local[['gf', 'pj']].apply(lambda x: (x.gf/x.pj)*media_goles_local, axis=1)
campaña_local['fdl'] = campaña_local[['gc', 'pj']].apply(lambda x: (x.gc/x.pj)/media_goles_visita, axis=1)
campaña_visita['fav'] = campaña_visita[['gf', 'pj']].apply(lambda x: (x.gf/x.pj) / media_goles_visita, axis=1)


# %%
def ingresar_valor(texto, minval, maxval, restriccion, mostrar_titulo):
    valor = 0
    ciclo = 1
    print()
    if mostrar_titulo is True:
        print("Seleccione equipo")
        print()
        print(tabla_equipos)
    while ciclo == 1:
        print(texto)
        try:
            valor = int(input())
            if valor < minval:
                print("Debe ingresar valores mayor o igual a {}".format(minval))
                continue
            elif valor > maxval:
                print("Debe ingresar valores menor o igual a {}".format(maxval))
            if valor == restriccion:
                print("El equipo ya fue seleccionado")
            else:
                ciclo = 0
        except Exception:
            print("Ingresar solo valores numéricos")
    return valor


# %%
def mostrar_valores(titulo, df):
    print()
    print(titulo)
    print(df)
    print()
    return


# %%
def calcular_probabilidad(factorlocal, factorvisita, goles):
    resultados = []
    for i in range(goles):
        x = ss.poisson.pmf(k=i, mu=factorlocal)
        y = ss.poisson.pmf(k=i, mu=factorvisita)
        z = (x*y)
        resultados.append([i, x, y, z])
    return pd.DataFrame(resultados, columns=['Goles', 'Prob local', 'Prob visita', 'Prov total'])


# %%
def sumas(df, columna, inicio):
    suma = 0
    while inicio <= len(df)-1:
        suma = suma + df.at[df.index[inicio], columna]
        inicio += 1
    return suma


# %%
def calcular_valores(df, colprob, colsuma):
    tope = 0
    valor = 0
    for i in range(len(df)):
        tp = df.at[df.index[i], colprob]
        tope = len(df) if tope >= len(df) else tope + 1
        sp = sumas(df, colsuma, tope)
        valor = valor + (tp*sp)
    return valor


# %%
def mostrar_probabilidad_partido(df):
    prob_local = calcular_valores(df, 'Prob visita', 'Prob local')
    prob_empate = df['Prov total'].sum()
    prob_visita = calcular_valores(df, 'Prob local', 'Prob visita')
    print()
    print("Probabilidad de resultados: ")
    print("Local: {:,.2f}%".format((prob_local*100)))
    print("Empate: {:,.2f}%".format((prob_empate*100)))
    print("Visita: {:,.2f}%".format((prob_visita*100)))
    return


# %%
def extraer_tabla(df, idlocal, idvisita):
    df1 = df.loc[df["ID"] == idlocal, ]
    df2 = df.loc[df["ID"] == idvisita, ]
    return pd.concat([df1, df2], axis=0)


# %%
def obtener_equipos(idlocal, idvisita, goles):
    tp_equipos = extraer_tabla(tabla_posiciones, idlocal, idvisita)
    cl_equipos = extraer_tabla(campaña_local, idlocal, idvisita)
    cv_equipos = extraer_tabla(campaña_visita, idlocal, idvisita)
    fal_local = cl_equipos.at[cl_equipos.index[0], 'fal']
    fdl_local = cl_equipos.at[cl_equipos.index[0], 'fdl']
    fal_visita = cl_equipos.at[cl_equipos.index[-1], 'fal']
    fdl_visita = cl_equipos.at[cl_equipos.index[-1], 'fdl']
    factor_local = (fal_local*fdl_visita)*media_goles_local
    factor_visita = (fal_visita*fdl_local)/media_goles_visita
    prob_goles = calcular_probabilidad(factor_local, factor_visita, goles)
    mostrar_valores("Tabla de posiciones", tp_equipos)
    mostrar_valores("Campaña de local", cl_equipos)
    mostrar_valores("Campaña de visita", cv_equipos)
    mostrar_probabilidad_partido(prob_goles)
    return


# %%
local = ingresar_valor("Ingrese id equipo local: ", 1, len(tabla_equipos), 0, True)
visita = ingresar_valor("Ingrese id equipo visita: ", 1, len(tabla_equipos), local, True)
goles = ingresar_valor("Ingrese máximo de goles del partido: ", 0, 9, 10, False)
obtener_equipos(local, visita, goles)
