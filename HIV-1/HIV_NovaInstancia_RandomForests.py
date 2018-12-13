# HIV-1 protease cleavage data
# https://archive.ics.uci.edu/ml/datasets/HIV-1+protease+cleavage
# Processamento de novas instãncias
# by Rodrigo Aires

# Atributos:
#   - Atributo String com 8 caracteres, onde cada caranter representa uma proteína.
#     Para cada posição será aceito um dos seguintes caranteres: ARNDCQEGHILKMFPSTWYV
#   - Atributo numérico que representa o rótulo de protease do HIV-1, onde 1 representa SIM e -1  representa NÃO


# Bibliotecas
import os
import pandas as pd
from sklearn.externals import joblib



# Abertura do arquivo de dados
file = 'files/NovaInstanciaHIV-1/746DataSemResult.txt'
columnsNameArquivo = ['Proteinas']
arquivo = pd.read_csv(file , delimiter=',',header=None)
arquivo.columns = columnsNameArquivo

print('Apresentando os primeiros registros do arquivo',file)
print(arquivo.head())
print('')


ColumnsName = ['P1','P2','P3','P4','P5', 'P6', 'P7', 'P8']
df = pd.DataFrame(columns=ColumnsName, dtype=str)


# Rotina utilizada para validar se as entradas estão de acordo com o padrão esperado.
# Caso negativo deve ser desconsiderado do teste
def ProteinasValidas(pLinha):
    ValoresValidos = 'ARNDCQEGHILKMFPSTWYV'
    if (len(pLinha) != 8) or \
            (ValoresValidos.find(pLinha[0]) == -1) or \
            (ValoresValidos.find(pLinha[1]) == -1) or \
            (ValoresValidos.find(pLinha[2]) == -1) or \
            (ValoresValidos.find(pLinha[3]) == -1) or \
            (ValoresValidos.find(pLinha[4]) == -1) or \
            (ValoresValidos.find(pLinha[5]) == -1) or \
            (ValoresValidos.find(pLinha[6]) == -1) or \
                (ValoresValidos.find(pLinha[7]) == -1):
        print('Entrada inválida: ',pLinha)
    else:
        return True

for index, row in arquivo.iterrows():
    linha = row['Proteinas']
    if (ProteinasValidas(linha)):
        reg = [{'P1': linha[0], 'P2': linha[1],'P3': linha[2],'P4': linha[3],'P5': linha[4],'P6': linha[5],'P7': linha[6],'P8': linha[7] }]
        df2 = pd.DataFrame(reg)
        # Concatenando os dataframes e retorno dos valores de HIV_1 igual à 0 para -1
        df = pd.concat([df, df2],ignore_index=True)

print('')
print('Apresentando os primeiros registros do Dataframe para análise')
print(df.head())
print('')


from sklearn.preprocessing import LabelEncoder
# Considerando que o algorítmo de aprendizagem não trabalha com atributos do tipo string. Existe a necessidade de realizar uma conversão de tipo.
# Para essa demanda, o LabelEncoder pode ser boa alternativa para atribuir um valor numérico para cada valor de do tipo string

print('Convertendo Data Frame em matriz')
X=df.apply(LabelEncoder().fit_transform).iloc[:,0:8].values.astype(str)
print(X)


# Acessando o modelo gravado no programa HIV_NovaInstancia_RandomForests.py
kmeans = joblib.load('HIV-1_RF.joblib')

# Descritivo PARA HIV-1
def DescPrevisao(i):
    switcher = {
        '1': 'Sim',
        '-1': 'Não',
    }
    return switcher.get(i, "previsão inválida")

# Percorrer a lista de novas entradas e apresentar as previsões com base no modelo gravado
for item in X:
    previsao = kmeans.predict([item])
    print(#"Nova Instância:", item, "- Precisão de clusters:",
          previsao, '-', DescPrevisao(previsao[0]))
