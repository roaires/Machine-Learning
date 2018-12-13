# HIV-1 protease cleavage data
# https://archive.ics.uci.edu/ml/datasets/HIV-1+protease+cleavage
# Geração de modelo de aprendizado
# by Rodrigo Aires


# Atributos:
#   - Atributo String com 8 caracteres, onde cada caranter representa uma proteína.
#     Para cada posição será aceito um dos seguintes caranteres: ARNDCQEGHILKMFPSTWYV
#   - Atributo numérico que representa o rótulo de protease do HIV-1, onde 1 representa SIM e -1  representa NÃO


# Bibliotecas
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split


# Abertura do arquivo de dados
file = 'files/newHIV-1_data/1625Data.txt'
#file = 'files/newHIV-1_data/schillingData.txt'
columnsNameArquivo = ['Proteinas','Label']
arquivo = pd.read_csv(file , delimiter=',',header=None)
arquivo.columns = columnsNameArquivo

print('Apresentando os primeiros registros do arquivo',file)
print(arquivo.head())
print('')

ColumnsName = ['P1','P2','P3','P4','P5', 'P6', 'P7', 'P8', 'HIV_1']
df = pd.DataFrame(columns=ColumnsName, dtype=str)


# Conhecendo as labels do dataset
print('Quantidade de registros por HIV_1 - Label')
agrupado = arquivo.groupby('Label').agg(['count'])
print(agrupado)
print('')


for index, row in arquivo.iterrows():
    # Optou-se em converter -1 para 0 visando facilitar a montagem do dataframe intermediário
    label = row['Label']
    if label < 0:
        label = 0
    linha = row['Proteinas'] + str(label)
    reg = [{'P1': linha[0], 'P2': linha[1],'P3': linha[2],'P4': linha[3],'P5': linha[4],'P6': linha[5],'P7': linha[6],'P8': linha[7],'HIV_1': linha[8]}]
    df2 = pd.DataFrame(reg)
    # Concatenando os dataframes e retorno dos valores de HIV_1 igual à 0 para -1
    df = pd.concat([df, df2],ignore_index=True).replace('0','-1')



print('Apresentando os primeiros registros do Dataframe para análise')
print(df.head())
print('')


print('Número de linhas e colunas',df.shape)
print('')

# Definição dos atributos e classes
# X : Atributo
# Y : Classes
X = df.drop(['HIV_1'], axis=1) # Removendo a coluna de classes HIV_1
y = df['HIV_1']

print('Apresentando os primeiros registros dos atributo')
print(X.head())
print('')

print('Apresentando os primeiros registros de classes')
print(y.head())
print('')

# Separar 20% dos registros para treinamento e e outros 20% para testes
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

# Considerando que o algorítmo de aprendizagem não trabalha com atributos do tipo string. Existe a necessidade de realizar uma conversão de tipo.
# Para essa demanda, o LabelEncoder pode ser boa alternativa para atribuir um valor numérico para cada valor de do tipo string
from sklearn.preprocessing import LabelEncoder

# Treinamento
classifier = RandomForestClassifier()
classifier.fit(X_train.apply(LabelEncoder().fit_transform), y_train)

# Testes
y_pred = classifier.predict(X_test.apply(LabelEncoder().fit_transform))


# #Valores previstos para o segmento de teste
print('Resultado do segmento de teste')
print(y_pred)
print('')


# #Avaliar o modelo: Acurácia e matriz de contingência
print("Resultado da Avaliação do Modelo")
print("Matriz de confusão")
print(confusion_matrix(y_test, y_pred))
print('')
print("Relatório de classificação")

print(classification_report(y_test, y_pred))
print('')

#Salvar o modelo para uso posterior
joblib.dump(classifier, 'HIV-1_RF.joblib')
print('Gerado modelo HIV-1_RF.joblib com sucesso!')

