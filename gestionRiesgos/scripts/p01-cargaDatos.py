# %%
import pandas as pd
import pandas_profiling
from pandas_profiling import ProfileReport
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from os import scandir, getcwd, path

# %%
def load_workbook(path_excel_file, file_name):
    excel_file = os.path.join(path_excel_file, file_name)
    return pd.ExcelFile(excel_file)

# %%
def load_worksheet(workbook, worksheet_name, init_row, drop_fields):
    worksheet = workbook.parse(sheet_name=worksheet_name, skiprows=init_row)
    worksheet.drop(drop_fields, axis=1, inplace=True)
    return pd.DataFrame(worksheet)

# %%
def combine_dataframe(df_left, df_right):
    df_merged = pd.merge(
        left=df_left, right=df_right, 
        left_on='ID', right_on='ID')
    return df_merged

# %%
def generate_dataframe(workbook):
    risk_list = load_worksheet(workbook, 'identificacionRiesgos', 4, ["Respuesta", "Causa raíz", "Identificación"])
    risk_matrix = load_worksheet(workbook, 'matriz', 4, ["Riesgo", "Tipo"])
    risk_response = load_worksheet(workbook, 'respuestaRiesgos', 4, ["Riesgo", "global", "Prioridad"])
    df = combine_dataframe(risk_list, risk_matrix)
    return  combine_dataframe(df, risk_response)

# %%
def clean_dataframe(df):
    df.drop(["A"], axis=1, inplace=True)
    df.columns = ["ID", "riesgo", "categoría", "subcategoría", "tipo_riesgo", "objetivo_alcance", 
    "objetivo_cronograma", "objetivo_costos", "objetivo_calidad", "tipo_impacto", "probabilidad", "valor_probabilidad", 
    "impacto_alcance", "impacto_cronograma", "impacto_costos", "impacto_calidad", "valor_alcance", "valor_cronograma", 
    "valor_costos", "valor_calidad", "puntuacion_alcance", "puntuacion_cronograma", "puntuacion_costos", "puntuacion_calidad", 
    "valoracion_global", "prioridad_riesgo", "dueño", "responsable", "plan_respuesta_predet", "estrategia_respuesta", 
    "plan_respuesta_adapt", "riesgo_residual", "riesgo_activado", "fecha_activacion"]
    return df



# %% [markdown]
# # Cargando datos de riesgos

# %%
data=pd.DataFrame({'A' : []})
path_excel_files ="../data/original"
file_list = [file.name for file in scandir(path_excel_files) if file.is_file()]
for file in file_list:
    workbook = load_workbook(path_excel_files, file)
    data_temp = generate_dataframe(workbook) 
    data = pd.concat([data, data_temp], axis=0)
data = clean_dataframe(data)
data 



# %% [markdown]
# # Visualizando estado del dataframe de riesgos

# %%
data.describe(include="all")

# %%
data.dtypes


# %% [markdown]
# # Transformando datos del dataframe de riesgos

# %%
def change_value_fields(df, fields):
    for field in fields:
        df.loc[df[field]=="X", field] = 1
        df.loc[df[field]=="-", field] = np.NAN
    return df

# %%
fields = ["objetivo_alcance", "objetivo_cronograma", "objetivo_costos", "objetivo_calidad",
"impacto_alcance", "impacto_cronograma", "impacto_costos", "impacto_calidad"] 
data = change_value_fields(data, fields)
#data.drop(["valor_probabilidad", "valor_alcance", "valor_cronograma", "valor_costos", 
#"valor_calidad", "puntuacion_alcance", "puntuacion_cronograma", "puntuacion_costos", "puntuacion_calidad"], axis=1, inplace=True)
data.head()

# %% [markdown]
# Categorizando los riesgos



# %%
data["proyecto"]=""
data.loc[data["ID"].str.contains("IN"), "proyecto"]="Intranet"
data.loc[data["ID"].str.contains("FM"), "proyecto"]="Farmacia"
data.loc[data["ID"].str.contains("IS"), "proyecto"]="Isapre"
data.loc[data["ID"].str.contains("VS"), "proyecto"]="Vida sana"
data.loc[data["ID"].str.contains("TM"), "proyecto"]="Telemedicina"
data.head()


# %% [markdown]
# # Visualizando estado de las variables

# %%
%matplotlib inline 
def show_plot(field, title, xtitle, ytitle):
    data[field].hist()
    plt.title(title)
    plt.xlabel(xtitle)
    plt.ylabel(ytitle)
    plt.show()
    return 



# %%
show_plot("proyecto", "Relación de tipos de proyectos", "Tipo proyecto", "Cantidad de riesgos")

# %%
show_plot("tipo_riesgo", "Relación de tipos de riesgos", "Tipo de riesgo", "Cantidad de riesgos")

# %%
show_plot("tipo_impacto", "Relación de tipos de impacto", "Tipo de impacto", "Cantidad de riesgos")

# %%
show_plot("probabilidad", "Relación de probabilidad de riesgo", "Probabilidad", "Cantidad de riesgos")

# %%
show_plot("prioridad_riesgo", "Relación de prioridad de riesgos", "Prioridad", "Cantidad de riesgos")

# %% [markdown]
# # Estructura de desglose de riesgos

# %%
rbs = data.pivot_table(index=["categoría", "subcategoría", "ID", "riesgo"], columns="ID", aggfunc="count")
rbs 




# %% [markdown]
# # Almacenando el dataframe de riesgos

# %%
data_file="../data/intermediate/riesgos.csv"
data.to_csv(data_file, index=False, encoding='utf-8-sig', float_format='%.2f')
print(data_file, " almacenado.")

# %%
fields = ["riesgo", "dueño", "responsable", "plan_respuesta_predet",
"estrategia_respuesta", "plan_respuesta_adapt", "riesgo_residual", "riesgo_activado", "fecha_activacion"]
data.drop(fields, axis=1, inplace=True)

# %%
data_file="../data/intermediate/riesgos-analisis.csv"
data.to_csv(data_file, index=False, encoding='utf-8-sig', float_format='%.2f')
print(data_file, " almacenado.")

# %%
profile = ProfileReport(data, title='Pandas Profiling Report')
profile