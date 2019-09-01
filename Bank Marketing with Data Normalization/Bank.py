
# Bibliotecas
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split



# Rotinas Normalização
def normalizarNumericos(dfOriginal,dfNormalizado,coluna):
    menorOriginal = min(dfOriginal[coluna])
    maiorOriginal = max(dfOriginal[coluna])
    dfNormalizado[coluna] = dfOriginal[coluna]
    for indice, linha in dfNormalizado.iterrows():
        dfNormalizado.loc[indice,coluna] = (linha[coluna]-menorOriginal)/(maiorOriginal-menorOriginal)

def normalizarNumericosNovos(dfOriginal, dfNovos,dfNormalizado,coluna):
    menorOriginal = min(dfOriginal[coluna])
    maiorOriginal = max(dfOriginal[coluna])
    dfNormalizado[coluna] = dfNovos[coluna]
    for indice, linha in dfNormalizado.iterrows():
        dfNormalizado.loc[indice,coluna] = (linha[coluna]-menorOriginal)/(maiorOriginal-menorOriginal)


def normalizarPDaysNovos(dfOriginal, dfNovos,dfNormalizado,coluna):
    dfNormalizado[coluna] = dfNovos[coluna]
    valores = dfOriginal[coluna].unique()
    menorOriginal = min(valores[1:])
    maiorOriginal = max(dfOriginal[coluna])
    for indice, linha in dfNormalizado.iterrows():
        if linha[coluna] != -1:
            dfNormalizado.loc[indice, coluna] = (linha[coluna] - menorOriginal) / (maiorOriginal - menorOriginal)

def normalizarPDays(dfOriginal,dfNormalizado,coluna):
    dfNormalizado[coluna] = dfOriginal[coluna]
    valores = dfOriginal[coluna].unique()
    menorOriginal = min(valores[1:])
    maiorOriginal = max(dfOriginal[coluna])
    for indice, linha in dfNormalizado.iterrows():
        if linha[coluna] != -1:
            dfNormalizado.loc[indice, coluna] = (linha[coluna] - menorOriginal) / (maiorOriginal - menorOriginal)



def normalizarDias(dfOriginal,dfNormalizado,coluna):
    fatorDia = 0.06667
    dfNormalizado[coluna] = dfOriginal[coluna]
    for indice, linha in dfNormalizado.iterrows():
        dfNormalizado.loc[indice,coluna] = (-1)+((linha[coluna]-1)*fatorDia)

def normalizarDummies(dfOriginal,dfNormalizado,listaColunas):
    for coluna in listaColunas:
        dfNormalizado[coluna] = dfOriginal[coluna]
    # durante os esperimentos foi necessário manter drop_first como False para manter os campos com sufixo yes e no.
    # a alteração desse parâmetro resulta na exclusão de um dos campos devido. Quanto ao prefixo, não foi incluso um "_"
    # para evitar que o campo ficasse nomeado com 2 "_" entre o nome o campo e o valor
    dfNormalizado = pd.get_dummies(dfNormalizado,
                                  columns=listaColunas,
                                  drop_first=False,
                                  prefix=listaColunas)
    return dfNormalizado

def normalizarJob(dfOriginal,dfNormalizado,coluna):
    listaJobs = ['unknown','unemployed','admin.','management','housemaid','entrepreneur','student','blue-collar','self-employed','retired','technician','services']
    # Definição do primeiro item como zero
    listaNormalizado = [0]
    # Normalizar do segundo ao últomo item
    fator = round(1/(len(listaJobs)-1),2)
    normalizado = 0
    for i in range(1,len(listaJobs)):
        normalizado = round(normalizado + fator,2)
        listaNormalizado.append(normalizado)

    dfNormalizado[coluna] = dfOriginal[coluna]
    for indice, linha in dfNormalizado.iterrows():
        dfNormalizado.loc[indice, coluna] = listaNormalizado[listaJobs.index(linha[coluna])]

def normalizarMonth(dfOriginal,dfNormalizado,coluna):
    listaMeses = ['jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec']
    # Definição do primeiro item como zero
    listaNormalizado = [-1]
    # Normalizar do segundo ao últomo item
    fator = 1/5.5
    normalizado = -1
    for i in range(1,len(listaMeses)):
        normalizado = round(normalizado + fator,3)
        listaNormalizado.append(normalizado)

    dfNormalizado[coluna] = dfOriginal[coluna]
    for indice, linha in dfNormalizado.iterrows():
        dfNormalizado.loc[indice, coluna] = listaNormalizado[listaMeses.index(linha[coluna])]



def normalizarMarital(dfOriginal,dfNormalizado,coluna):
    listaMarital = ['single','divorced','married']
    listaNormalizado = [-1,0,1]
    dfNormalizado[coluna] = dfOriginal[coluna]
    for indice, linha in dfNormalizado.iterrows():
        dfNormalizado.loc[indice, coluna] = listaNormalizado[listaMarital.index(linha[coluna])]


def normalizarCrescente4Itens(dfOriginal,dfNormalizado,coluna, listaValores):
    listaNormalizado = [0,0.33,0.66,1]
    dfNormalizado[coluna] = dfOriginal[coluna]

    for indice, linha in dfNormalizado.iterrows():
        dfNormalizado.loc[indice, coluna] = listaNormalizado[listaValores.index(linha[coluna])]


def normalizarPoutcome(dfOriginal,dfNormalizado,coluna):
    listaNormalizado = [-1,0,0.5,1]
    listaValores = ['failure','unknown','other','success']
    dfNormalizado[coluna] = dfOriginal[coluna]

    for indice, linha in dfNormalizado.iterrows():
        dfNormalizado.loc[indice, coluna] = listaNormalizado[listaValores.index(linha[coluna])]

def normalizarContact(dfOriginal,dfNormalizado,coluna):
    listaNormalizado = [-1,0,1]
    listaValores = ['telephone','unknown','cellular']
    dfNormalizado[coluna] = dfOriginal[coluna]

    for indice, linha in dfNormalizado.iterrows():
        dfNormalizado.loc[indice, coluna] = listaNormalizado[listaValores.index(linha[coluna])]

def normalizarCopiarValoresClasse(dfOriginal,dfNormalizado,coluna):
    dfNormalizado[coluna] = dfOriginal[coluna]


# Abertura do arquivo de dados
file = 'data/bank.csv'
columnsNameArquivo = ['age','job','marital','education','default','balance','housing','loan','contact','day','month',
                      'duration','campaign','pdays','previous','poutcome','subscribe']

arquivo = pd.read_csv(file , delimiter=';',header=0)
arquivo.columns = columnsNameArquivo

print('Apresentando os primeiros registros do arquivo',file)
print(arquivo.head())
print('')

# Exibe o número de linhas e o número de colunas
print(arquivo.shape)

dfNormalizado = pd.DataFrame(columns=columnsNameArquivo)

# Normalizações
normalizarNumericos(arquivo,dfNormalizado, 'age')
normalizarNumericos(arquivo,dfNormalizado, 'balance')
normalizarNumericos(arquivo,dfNormalizado, 'duration')
normalizarNumericos(arquivo,dfNormalizado, 'campaign')
normalizarNumericos(arquivo,dfNormalizado, 'previous')
normalizarDias(arquivo,dfNormalizado, 'day')
normalizarJob(arquivo,dfNormalizado, 'job')
normalizarMarital(arquivo,dfNormalizado, 'marital')
normalizarCrescente4Itens(arquivo,dfNormalizado, 'education', ['unknown','primary', 'secondary','tertiary'])
normalizarPoutcome(arquivo,dfNormalizado, 'poutcome')
normalizarContact(arquivo,dfNormalizado, 'contact')
normalizarPDays(arquivo,dfNormalizado, 'pdays')
normalizarMonth(arquivo,dfNormalizado, 'month')
normalizarCopiarValoresClasse(arquivo,dfNormalizado, 'subscribe')
dfNormalizado = normalizarDummies(arquivo,dfNormalizado,['default','housing', 'loan'])

df = dfNormalizado

print('Apresentando os primeiros registros do Dataframe para análise')
print(df.head(30))
print('')


print('Número de linhas e colunas',df.shape)
print('')



# Definição dos atributos e classes
# X : Atributo
# Y : Classes
y = df['subscribe']
X = df.drop(['subscribe'], axis=1) # Removendo a coluna de classes subscribe


print('Apresentando os primeiros registros dos atributo')
print(X.head())
print('')

print('Apresentando os primeiros registros de classes')
print(y.head())
print('')

# Separar 20% dos registros para treinamento e e outros 20% para testes
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)



# Treinamento
# Na base de dados proposta, a opção de balanceamento de classes não demonstrou grandes alterações e melhorias na previsão da classe "yes"
# Quando a classe "no" manteve-se com uma precisão interessante
classifier = RandomForestClassifier(class_weight='balanced')
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
joblib.dump(classifier, 'data/bank_RF.joblib')
print('Gerado modelo bank_RF.joblib com sucesso!')
print('')
print('')


print('### Trabalhando com novas instâncias ###')


# Abertura do arquivo de dados
file = 'data/bank-full.csv'
columnsNameArquivo = ['age','job','marital','education','default','balance','housing','loan','contact','day','month',
                      'duration','campaign','pdays','previous','poutcome','subscribe']

arquivoNovos = pd.read_csv(file , delimiter=';',header=0)
arquivoNovos.columns = columnsNameArquivo
# Considerar apenas os 100 primeiros registros como novas intânciadas
arquivoNovos = arquivoNovos.head(100)


print('Apresentando os primeiros registros do arquivo',file)
print(arquivoNovos.head())
print('')


# Exibe o número de linhas e o número de colunas
print(arquivoNovos.shape)



dfNormalizado = pd.DataFrame(columns=columnsNameArquivo)

# Normalizações
normalizarNumericosNovos(arquivo,arquivoNovos,dfNormalizado, 'age')
normalizarNumericosNovos(arquivo,arquivoNovos,dfNormalizado, 'balance')
normalizarNumericosNovos(arquivo,arquivoNovos,dfNormalizado, 'duration')
normalizarNumericosNovos(arquivo,arquivoNovos,dfNormalizado, 'campaign')
normalizarNumericosNovos(arquivo,arquivoNovos,dfNormalizado, 'previous')
normalizarDias(arquivoNovos,dfNormalizado, 'day')
normalizarJob(arquivoNovos,dfNormalizado, 'job')
normalizarMarital(arquivoNovos,dfNormalizado, 'marital')
normalizarCrescente4Itens(arquivoNovos,dfNormalizado, 'education', ['unknown','primary', 'secondary','tertiary'])
normalizarPoutcome(arquivoNovos,dfNormalizado, 'poutcome')
normalizarContact(arquivoNovos,dfNormalizado, 'contact')
normalizarPDaysNovos(arquivo,arquivoNovos,dfNormalizado, 'pdays')
normalizarMonth(arquivoNovos,dfNormalizado, 'month')
normalizarCopiarValoresClasse(arquivoNovos,dfNormalizado, 'subscribe')
dfNormalizado = normalizarDummies(arquivoNovos,dfNormalizado,['default','housing', 'loan'])

df = dfNormalizado

print('Apresentando os primeiros registros do Dataframe para análise')
print(df.head(30))
print('')


print('Número de linhas e colunas',df.shape)
print('')

print('Colunas normalizadas do arquivo de novas instâncias',df.columns)
print('')


# Armazenar resultado esperado com base no arquivo de novas instâncias

y = df['subscribe']
X = df.drop(['subscribe'], axis=1) # Removendo a coluna de classes subscribe


print(X)

# convertendo df em lista para passagem da instância de forma individualizada
listaNovaInstancia = X.values.tolist()

print("Validando novas instâncias")
for i in range(0, len(listaNovaInstancia)):
    novaInstancia = listaNovaInstancia[i]
    pred = classifier.predict([novaInstancia])[0]
    linha = i+1
    print('Linha:', str(linha), '' ,'Esperado:',y[i],'' , 'Previsão Modelo:', pred)
