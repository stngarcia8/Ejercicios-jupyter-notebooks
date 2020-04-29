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
Normalizando nombres de columnas

# %%
data.columns=[
    "Id", "Elevation", "Aspect", "Slope", 
    "H_Hydrology", "V_Hydrology", "H_Roadways", 
    "sh_9AM", "sh_Noon", "sh_3PM", 
    "H_Fire_Points", "Wilderness1", "Wilderness2", "Wilderness3", "Wilderness4",
    "Soil1", "Soil2", "Soil3", "Soil4", "Soil5", "Soil6", "Soil7", "Soil8", 
    "Soil9", "Soil10", "Soil11", "Soil12", "Soil13", "Soil14", "Soil15", "Soil16", 
    "Soil17", "Soil18", "Soil19", "Soil20", "Soil21", "Soil22", "Soil23", "Soil24", 
    "Soil25", "Soil26", "Soil27", "Soil28", "Soil29", "Soil30", "Soil31", "Soil32", 
    "Soil33", "Soil34", "Soil35", "Soil36", "Soil37", "Soil38", "Soil39", "Soil40",
    "Cover_Type"]
data.columns


# %% [markdown]
# ## normalizando columnas categoricas

# %%
def quitar_dummies(df, lista_cols, result_col):
    df_temp = data[lista_cols]
    df_temp = df_temp.set_index(["Id"])
    x = df_temp.stack() 
    var = pd.Series(pd.Categorical(x[x!=0].index.get_level_values(1)))
    df_temp = pd.concat([df_temp, var], axis=1)
    lista_cols.append(result_col)
    df_temp.reset_index(level=0, inplace=True)
    df_temp.columns=[lista_cols]
    lista_cols.pop(-1)
    df_temp.drop(lista_cols, axis=1, inplace=True)
    return df_temp


# %% [markdown]
# ## Tratamiento a columna wildness_area

# %%
copied_data = data.copy()
cols=["Id", "Wilderness1", "Wilderness2", "Wilderness3", "Wilderness4"]
df_warea = quitar_dummies(copied_data, cols, "Wilderness_Area")
cols.pop(0)
copied_data.drop(cols, axis=1, inplace=True)
new_data = pd.concat([copied_data, df_warea], axis=1)
new_data.head(15)


# %% [markdown]
# ## Tratamiento a columnas Soil_type

# %%
copied_data = new_data.copy()
cols=["Id",
        "Soil1", "Soil2", "Soil3", "Soil4", "Soil5", "Soil6", "Soil7", "Soil8", 
        "Soil9", "Soil10", "Soil11", "Soil12", "Soil13", "Soil14", "Soil15", "Soil16", 
        "Soil17", "Soil18", "Soil19", "Soil20", "Soil21", "Soil22", "Soil23", "Soil24", 
        "Soil25", "Soil26", "Soil27", "Soil28", "Soil29", "Soil30", "Soil31", "Soil32", 
        "Soil33", "Soil34", "Soil35", "Soil36", "Soil37", "Soil38", "Soil39", "Soil40"]
df_warea = quitar_dummies(copied_data, cols, "Soil_Type")
cols.pop(0)
copied_data.drop(cols, axis=1, inplace=True)
new_data = pd.concat([copied_data, df_warea], axis=1)
new_data.head(15)



# %% [markdown]
# ## Reasignando nombres de columnas

# %%
new_data.columns=[
    "Id", "Elevation", "Aspect", "Slope", 
    "H_Hydrology", "V_Hydrology", "H_Roadways", 
    "sh_9AM", "sh_Noon", "sh_3PM", 
    "H_Fire_Points", "Cover_Type", "Wilderness_Area", "Soil_Type"]
new_data.columns


# %% [markdown]
# ## Grabando el dataset limpio (intermedio)

# %%
data_file="../data/intermediate/forest-limpio.csv"
new_data.to_csv(data_file, index=False)
print(data_file, " almacenado.")

# %% [markdown]
# # Limpieza de dataset Forest concluida.