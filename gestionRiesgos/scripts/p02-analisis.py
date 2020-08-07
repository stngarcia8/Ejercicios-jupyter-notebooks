# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# %% [markdown]
# # Analisis del proyecto Vida Sana

# %%
data_file="../data/intermediate/riesgos-analisis.csv"
data = pd.read_csv(data_file)
data.shape 

# %%
def generate_pie_legend(df):
    df = df.reset_index(level=0)
    df.columns = ['field', 'count']
    df['result'] = df['field'] + ' (' + df['count'].astype(str) + ')'
    df.drop(['field', 'count'], axis=1, inplace=True)
    return df['result']

# %%
def generate_bar_legend(df, field, subfield):
    df = df.groupby(field)[subfield].value_counts().to_frame(name = 'count').reset_index() 
    df['result'] = df[field] + ' ' + df[subfield] + ' (' + df['count'].astype(str) + ')'
    df.drop([field, subfield, 'count'], axis=1, inplace=True)
    return df['result']


# %%
%matplotlib inline
def pie_chart(title, field, colors):
    labels = data[field].unique()
    explode = [0.1 for i in range(len(labels))] 
    explode[-1]=0
    explode = tuple(explode)
    fig1, ax1 = plt.subplots(1, 1, figsize=(13, 6))
    data[field].value_counts().plot(kind="pie", legend=False, autopct='%1.1f%%',shadow=True, startangle=90, explode=explode, colors=colors)
    plt.title(title)
    plt.ylabel('')
    fig1.legend(generate_pie_legend(data[field].value_counts()), loc='center right', borderaxespad=0.1, title=field.capitalize().replace('_', ' '))
    plt.subplots_adjust(right=0.85)
    plt.show()
    return

# %%
%matplotlib inline
def bar_chart(title, field, subfield, xlabel, ylabel, direction, colors):
    fig1, ax1 = plt.subplots(1, 1, figsize=(13, 6))
    if direction == None:
        d = sns.countplot(x=field, data=data, hue=subfield, palette=colors)
    else:
        d = sns.countplot(y=field, data=data, hue=subfield, palette=colors)
    d.legend(loc='best')
    plt.setp(d.get_xticklabels(), rotation=0)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.show()
    return



# %% [markdown]
# ## Distribución de riesgos por proyectos

#%%
pie_chart('Distribución de riesgos por proyecto', 'proyecto', None)
data['proyecto'].value_counts()



# %% [markdown]
# ## Distribución de amenazas y oportunidades en el proyecto

# %%
pie_chart('Distribución de tipos de riesgos', 'tipo_riesgo', ['Red', 'Green'])
data['tipo_riesgo'].value_counts()



# %% [markdown]
# ## Distribución de probabilidades de riesgos en el proyecto

# %%
pie_chart('Distribución de prioridades de riesgos', 'prioridad_riesgo', ['Yellow', 'Green', 'Red'])
data['prioridad_riesgo'].value_counts()



# %% [markdown]
# ## Distribución de categorías de riesgos

# %%
pie_chart('Distribución de categorías de riesgos', 'categoría', None)
data['categoría'].value_counts()



# %% [markdown]
# ## Distribución de subcategorías de riesgos

# %%
fig=plt.subplots(figsize=(20,6))
d=sns.countplot(x='subcategoría',data=data,palette='bright')
m=plt.setp(d.get_xticklabels(),rotation=75)
plt.title('Distribución de subcategorías de riesgos según su clasificación')
plt.ylabel("Cantidad de riesgos")
plt.xlabel("Proyecto")
plt.show()
data['subcategoría'].value_counts()



# %% [markdown]
# ## Distribución de amenazas y oportunidades por proyecto

# %%
bar_chart('Relación de amenazas y oportunidades por proyecto', 'proyecto', 'tipo_riesgo', 'Cantidad de riesgos', 'Proyecto', 'H', ['red', 'green'])
data.groupby('proyecto')['tipo_riesgo'].value_counts()



# %% [markdown]
# ## Distribución de prioridade de riesgo por proyecto

# %%
bar_chart('Relación de prioridad de riesgo por proyecto', 'proyecto', 'prioridad_riesgo', 'Proyecto', 'Cantidad de riesgos', None, ['yellow', 'green', 'red'])
data.groupby('proyecto')['prioridad_riesgo'].value_counts()




# %% [markdown]
# ## Distribución de categorías de riesgos por proyecto

# %%
bar_chart('Relación de categorías de riesgos por proyecto', 'proyecto', 'categoría', 'Proyecto', 'Cantidad de riesgos', None, None)
data.groupby('proyecto')['categoría'].value_counts()

