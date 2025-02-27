import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.linear_model import LinearRegression
import statsmodels.formula.api as sm
import numpy as np 
from yellowbrick.regressor import ResidualsPlot
dados = pd.read_csv('mt_cars.csv')
# dados.drop  Remove colunas ou linhas de um DataFrame. 
# ['Unamed: 0']  Nome da coluna que será removida.
# axis=1  Especifica que a remoção será feita em colunas (se fosse axis=0, removeria uma linha).
dados = dados.drop(['Unnamed: 0'], axis=1)

print(dados.head())
#Conta quantas linhas e quantas colunas há no dataset
print(dados.shape)
# variavel dependente y:
# variável é dependente, pois é o alvo da predição.
# variavel independente x:
# variável é independente, ou seja, o modelo a usa para prever y.
#A relação entre x e y geralmente segue a equação:
#y=a⋅x+b
# (a )é o coeficiente angular (inclinação) e (b) é o intercepto.
x = dados.iloc[:, 2].values # coluna disp(cilindrada)
y = dados.iloc[:, 0].values # coluna mpg(consumo)
print(x)
# verificar a correlação entre as variaveis
correlacao = np.corrcoef(x, y) 
# Correlação Negativa forte, correlação = -0.84755138
# Enquanto uma cresce a outra decresce
print(f'Correlação entre x e y: {correlacao}')

# formato de matriz com uma coluna a mais.
# -1: O parâmetro -1 diz ao Python para calcular automaticamente o número de linhas,
# com base no tamanho total da matriz. Ou seja, 
# ele se adapta à quantidade de elementos em x.
# 1: O valor 1 indica que queremos que x tenha uma coluna.
x = x.reshape(-1, 1)
# Criação do modelo e treinamento
# (fit indica que o treinamento deve ser iniciado)
modelo = LinearRegression()
#treina o modelo de regressão linear
modelo.fit(x, y)

#Visualização do coeficiente
#Interceptação
# onde os dados encontram o eixo y
print(f'Interceptação: {modelo.intercept_}')

#Inclinação
#o quanto a minha variavel y cresce conforme a x cresce
#angulo da reta
print(f'Coeficiente{modelo.coef_}')

# calculo de R^2, coeficiente de determinação
print(f'R^2: {modelo.score(x,y)}')

# Geração de Previsões
print(f'Previsão: {modelo.predict(x)}')

#Grafico com os pontos reais e as previsões
plt.scatter(x, y)
plt.plot(x, modelo.predict(x), color='red')
plt.show()


print(f'Previsão de um automovel que tem 200 cilindradas, quantas milhas fará: {modelo.predict([[200]])}')

#Grafico para visualizar os Residuos
#Verifica se temos um bom modelo
visualizador = ResidualsPlot(modelo)
# entender o relacionamento entre os dados para gerar visualizações úteis
visualizador.fit(x,y)
#exibir a visualização 
visualizador.poof()

#R² = 0.718 significa que 71,8% da variabilidade em y (mpg) 
# pode ser explicada pela variação de x  no modelo de regressão linear.
#O restante (28.2%) pode ser devido a outros fatores não capturados pelo modelo ou à variabilidade aleatória.


#----------------------------------------------------------------------------------------------------------------
#                                   REGRESSAO MULTIPLA

# Criação de modelo usando a biblioteca statsmodel
#Variavel dependete a esquerda 'mpg' e variavel independente a direita 'disp'
modelo_ajustado = sm.ols(formula= 'mpg ~ disp', data=dados)
modelo_treinado = modelo_ajustado.fit()
#Descrição do modelo treinado com valor de R^2, R ajustado, coeficiente e interceptação
print(modelo_treinado.summary())

#O Least Squares (ou Mínimos Quadrados, 
# em português) é um método matemático utilizado para encontrar a linha (ou hiperplano, 
# no caso de mais de uma variável independente) 
# que melhor se ajusta aos dados em um modelo de regressão. 
# Ele é amplamente usado em regressão linear para minimizar
# a soma dos erros quadráticos entre os valores observados 
# e os valores previstos pelo modelo.

# Criação de uma nova variavel e novo modelo para comparação com a anterior
# 3 variaveis dependentes para prever mpg: cyl, disp, hp
x1 = dados.iloc[:, 1:4].values # coluna cyl, disp, hp, 
y1 = dados.iloc[:, 0].values # coluna mpg(consumo)
print(x1)
modelo2 = LinearRegression()
modelo2.fit(x1, y1)
# R^2, coeficiente de determinação
print(f'R^2: {modelo2.score(x1, y1)}')

# Modelo usando a biblioteca statsmodels com 3 variavesis dependente
modelo_ajustado2 = sm.ols(formula= 'mpg ~ cyl + disp + hp', data=dados)
#Treina o modelo
modelo_treinado2 = modelo_ajustado2.fit()
#Descrição do modelo treinado com valor de R^2, R ajustado, coeficiente e interceptação
print(modelo_treinado2.summary())


# Logo um modelo com mais variaveis independentes obteve R-ajustado de  0.743, 
# enquanto o modelo com uma variavel independente obteve R ajustado de 0.709,
# Conclusão que o modelo com mais variaveis independentes obteve um melhor resultado.

# np.array([cyl, dips, hp])
#Previsão de um novo registro
previsao2 = np.array([4, 200, 100])
previsao2 = previsao2.reshape(1, -1)
print(f'Previsão: {modelo2.predict(previsao2)}')