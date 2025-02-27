import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.linear_model import LinearRegression
import numpy as np 
from yellowbrick.regressor import ResidualsPlot
dados = pd.read_csv('cars.csv')
print(dados.head())
# dados.drop  Remove colunas ou linhas de um DataFrame. 
# ['Unamed: 0']  Nome da coluna que será removida.
# axis=1  Especifica que a remoção será feita em colunas (se fosse axis=0, removeria uma linha).
dados = dados.drop(['Unnamed: 0'], axis=1)
#Boxplot para identificar outliers
sb.boxplot(dados)
plt.show()

print(dados['dist'])

print(dados.head())
# variavel dependente x (speed):
# variável é dependente, pois é o alvo da predição.
# variavel independente y (dist):
# variável é independente, ou seja, o modelo a usa para prever y.
#A relação entre x e y geralmente segue a equação:
#y=a⋅x+b
# (a )é o coeficiente angular (inclinação) e (b) é o intercepto.
x = dados.iloc[:, 1].values
y = dados.iloc[:, 0].values
print(x)
# verificar a correlação entre as variaveis
correlacao = np.corrcoef(x, y)
print(f'Correlação entre x e y: {correlacao}')

# formato de matriz com uma coluna a mais
x = x.reshape(-1, 1)
# Criação do modelo e treinamento
# (fit indica que o treinamento deve ser iniciado)
modelo = LinearRegression()
#treina o modelo de regressão linear
modelo.fit(x, y)

#Visualização do coeficiente
#Interceptação
print(f'Interceptação: {modelo.intercept_}')

#Inclinação
#o quanto a minha variavel y cresce conforme a x cresce
print(f'Coeficiente{modelo.coef_}')


#Grafico com os pontos reais e as previsões
plt.scatter(x, y)
plt.plot(x, modelo.predict(x), color='red')
plt.show()


# Previsão de distancia de 22 pés usando a formula manual:
# Iterceptação + inclinação * valor de dist
# Qual velocidade se levou 22 pés para parar
prev = modelo.intercept_ + modelo.coef_ * 22
print(f'Calculo manual: {prev}')
# Prveisão usando função do Sklearn
print(f'Previsão: {modelo.predict([[22]])}')
#Grafico para visualizar os Residuos
#Verifica se temos um bom modelo
visualizador = ResidualsPlot(modelo)
# entender o relacionamento entre os dados para gerar visualizações úteis
visualizador.fit(x,y)
#exibir a visualização 
visualizador.poof()


#R² = 0.651 significa que 65.1% da variabilidade em y (distancia) 
# pode ser explicada pela variação de x (velocidade) no modelo de regressão linear.
#Em outras palavras, 65.1% da variação dos dados de distancia
# é explicada pela velocidade. O restante (34.9%) 
# pode ser devido a outros fatores não capturados pelo modelo ou à variabilidade aleatória.