# Bibliotecas
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Datasets
train = pd.read_csv('titanic/train.csv')
test = pd.read_csv('titanic/test.csv')

train.set_index('PassengerId', inplace=True)
test.set_index('PassengerId', inplace=True)

dados = pd.concat([train, test], sort=False)

dados.columns = ['Sobreviveu', 'Classe', 'Nome', 'Sexo', 'Idade', 'IrmaosConjugues', 'PaisFilhos', 'Bilhete',
       'Tarifa', 'Cabine', 'Embarque']
dados['Sexo'] = dados['Sexo'].map({'male': 'homem', 'female': 'mulher'})

dados.info() ## Analisar as informações dos nossos dados

# Categórico Nominal, Sexo, Nome, Embarque, Sobreviveu
# Categórico Ordinal: Classe
# Contínuos: Idade, Tarifa
# Discretos: PaisFilhos, IrmaosConjuges
# Alfanumérico: Bilhete

dados.isnull().sum()
moda_embarque = dados['Embarque'].mode()[0]
dados.fillna({'Embarque': moda_embarque}, inplace=True)

# Informações Gerais
dados.describe()

# Informações da Amostra
len(dados)
len(train)/2224 ## Proporção total da amostra em relação á população.

1-1502/2224 ## Proporção de sobreviventes(População)

train['Survived'].value_counts()
342/len(train) ## Proporção de sobreviventes amostra

## Agrupamentos -- Classe x Sobreviveu
dados[['Classe', 'Sobreviveu']].groupby(['Classe'])\
    .mean().sort_values(by='Sobreviveu', ascending = False)

## Primeira classe muito mais propensa a sobrevier

## Agrupamento -- Sexo x Sobreviveu
dados[['Sexo', 'Sobreviveu']].groupby(['Sexo'])\
    .mean().sort_values(by='Sobreviveu', ascending = False)
    
## Mulheres sobreviveram muito mais

## Agrupamento -- IrmaosConjuges x Sobreviveu
dados[['IrmaosConjugues', 'Sobreviveu']].groupby(['IrmaosConjugues'])\
    .mean().sort_values(by='Sobreviveu', ascending = False)
    
## Agrupamento -- PaisFilhos x Sobreviveu
dados[['PaisFilhos', 'Sobreviveu']].groupby(['PaisFilhos'])\
    .mean().sort_values(by='Sobreviveu', ascending = False)
    
# Correlação de IrmaosConjuges e PaisFilhos
dados[['IrmaosConjugues', 'PaisFilhos']].corr()

# Visualização dos dados -- GRÁFICOS
## Pizza - Sobreviventes
f,ax=plt.subplots(1,2,figsize=(10,5))
dados['Sobreviveu'].value_counts().plot.pie(explode=[0,0.05],autopct='%0.2f%%',ax=ax[0])
ax[0].set_title('Sobreviveu')
ax[0].set_ylabel('')
sns.countplot(x='Sobreviveu', data = dados, palette=['blue', 'orange'], ax=ax[1])
ax[1].set_title('Sobreviveu')
ax[1].set_ylabel('')
plt.show()

## Barra - Sobreviventes por sexo  
dados.groupby(['Sexo','Sobreviveu'])['Sobreviveu'].count()
dados.loc[dados['Sexo']=='mulher']['Sobreviveu'].value_counts()
sns.countplot(x ='Sexo',hue='Sobreviveu',data=dados)
plt.show()

## Histograma - Idade X Sobreviveu
g = sns.FacetGrid(dados, col='Sobreviveu')
g.map(plt.hist, 'Idade', bins=18)
plt.show()

## Histograma - Idade X Classe X Sobreviveu com FacetGrid
grid = sns.FacetGrid(dados, col='Sobreviveu', row='Classe', height=2.2, aspect=1.6)
grid.map(plt.hist, 'Idade', alpha=0.7, bins=20)
grid.add_legend()
plt.show()

## Chances de Sobrevivência por Porto de Embarque
sns.catplot(x ='Embarque',y = 'Sobreviveu',data=dados, kind='point')
fig=plt.gcf()
fig.set_size_inches(6,3)
plt.show()

## Embarque X Classe X Sobreviveu
grid = sns.FacetGrid(data, row='Embarque', height=2.2, aspect=1.6)
grid.map(sns.pointplot, 'Classe', 'Sobreviveu', 'Sexo', palette='deep')
grid.add_legend()
plt.show()

## Agrupar mulheres e homens por sobrevivência
dados.loc[dados['Sexo']=='mulher'].groupby('Sobreviveu').mean()
dados.loc[dados['Sexo']=='homem'].groupby('Sobreviveu').mean()

## Barras - Tarifa por Sexo
sns.barplot(x = 'Sexo', y= 'Tarifa',hue='Sobreviveu',data=dados)
plt.show()
