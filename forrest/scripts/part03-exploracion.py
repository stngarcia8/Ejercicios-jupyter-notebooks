# %%
%load_ext watermark
%watermark


# %% [markdown]
# # Exploración de datos de Forest


# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# %%
data_file="../data/intermediate/forest-categorizado.pkl"
data_origen = pd.read_pickle(data_file)



# %%
data_origen.describe(include="all")

# %%
data_origen.info()

# %%
df_correlacion = data_origen.corr()
df_correlacion


# %% [markdown]
# ## Histograma de datos de forest

# %%
data_origen.hist()

# %% [markdown]
# ## Analisis de altura de los bosques

# %%
%matplotlib inline 
k = int(np.ceil(1+np.log2(len(data_origen))))
plt.hist(data_origen["Elevation"], bins = k) 
plt.xlabel("Altura de los bosques (metros)")
plt.ylabel("Ocurrencias")
plt.title("Histograma altura de los bosques")
plt.show()













# %%
.hist()
plt.title("Relación de áreas silvestres registradas")
plt.xlabel("Tipo área silvestre")
plt.ylabel("Ocurrencias")
plt.show()


# %%
%matplotlib inline 
data_origen.plot(kind="scatter", x="Elevation", y="Wilderness_Area")
plt.title("Relación de áreas silvestres vs altura")
plt.xlabel("Altura (metros")
plt.ylabel("Tipo área silvestre")
plt.show()


# %% [markdown]
# ## Analizando variable tipo de covertura

# %%
%matplotlib inline
data_origen["Cover_Type"].hist()