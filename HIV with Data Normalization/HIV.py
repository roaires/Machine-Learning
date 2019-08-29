# HIV-1 protease cleavage data
# https://archive.ics.uci.edu/ml/datasets/HIV-1+protease+cleavage
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



def retornarDFColunasP(arquivo) :
    # Separando cada proteína em uma coluna para trabalhar com os dados
    ColumnsName = ['P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'HIV_1']
    df = pd.DataFrame(columns=ColumnsName, dtype=str)

    for index, row in arquivo.iterrows():
        linha = row['Proteinas'] + str(row['Label'])
        reg = [{'P1': linha[0], 'P2': linha[1], 'P3': linha[2], 'P4': linha[3], 'P5': linha[4], 'P6': linha[5],
                'P7': linha[6], 'P8': linha[7], 'HIV_1': linha[8]}]
        df2 = pd.DataFrame(reg)
        # por algum motivo considerou apenas o "-" no -1, por isso o replace para "arrumar"
        df = pd.concat([df, df2], ignore_index=True).replace('-', '-1')
    return df

def normalizar(df):
    # Listagem de colunas para o df
    proteinasPossiveis = 'ARNDCQEGHILKMFPSTWYV'
    listaColunas = []
    for i in range(1, 9):
        for p in proteinasPossiveis:
            listaColunas.append('P' + str(i) + '_' + p)

    # Cria atributos "dummies" para as colunas que não são numericas no conjunto de dados
    # Estratégia utilizada pois a seqência da proteína pode afetar o resultado
    df2 = pd.get_dummies(df,
                          columns=['P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8'],
                          drop_first=True, prefix=['P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8'])

    df3 = pd.DataFrame(columns=listaColunas)
    for c in df2.columns:
        df3[c] = df2[c]
    return df3.fillna(0)


### Geração de modelo de aprendizado

# Abertura do arquivo de dados
file = 'data/746Data.txt'
columnsNameArquivo = ['Proteinas','Label']
arquivo = pd.read_csv(file , delimiter=',',header=None)
arquivo.columns = columnsNameArquivo

print('Apresentando os primeiros registros do arquivo',file)
print(arquivo.head())
print('')

# Exibe o número de linhas e o número de colunas
print(arquivo.shape)


df = retornarDFColunasP(arquivo)

print('Apresentando os primeiros registros do Dataframe para análise')
print(df.head(30))
print('')


print('Número de linhas e colunas',df.shape)
print('')

new_attributes = normalizar(df)


# Definição dos atributos e classes
# X : Atributo
# Y : Classes
X = new_attributes.drop(['HIV_1'], axis=1) # Removendo a coluna de classes HIV_1
y = df['HIV_1']

print('Apresentando os primeiros registros dos atributo')
print(X.head())
teste = X.head(1)
teste.values.tolist()
print('')

print('Apresentando os primeiros registros de classes')
print(y.head())
print('')

# Separar 20% dos registros para treinamento e e outros 20% para testes
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

# Treinamento
classifier = RandomForestClassifier()
classifier.fit(X_train, y_train)

# Testes
y_pred = classifier.predict(X_test)


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


### Trabalhando com novas instâncias
# Abertura do arquivo de dados

file = 'data/1625Data.txt'
columnsNameArquivo = ['Proteinas','Label']
arquivo = pd.read_csv(file , delimiter=',',header=None)
arquivo.columns = columnsNameArquivo

print('Apresentando os primeiros registros do arquivo',file)
print(arquivo.head())
print('')

# Exibe o número de linhas e o número de colunas
print(arquivo.shape)

df = retornarDFColunasP(arquivo)


# Abertura do modelo gravado
classifier = joblib.load('HIV-1_RF.joblib')

new_attributes = normalizar(df)

# Armazenar resultado esperado com base no arquivo de novas instâncias
y = new_attributes['HIV_1']

new_attributes = new_attributes.drop(['HIV_1'], axis=1) # Removendo a coluna de classes HIV_1
print(new_attributes)

# convertendo df em lista para passagem da instância de forma individualizada
listaNovaInstancia = new_attributes.values.tolist()

print("Validando novas instâncias")
for i in range(0, len(listaNovaInstancia)):
    novaInstancia = listaNovaInstancia[i]
    pred = classifier.predict([novaInstancia])[0]
    linha = i+1
    print('Linha:', str(linha), '','Nova Instância:', arquivo.loc[i, 'Proteinas'],'' ,'Esperado:',y[i],'' , 'Previsão Modelo:', pred)
