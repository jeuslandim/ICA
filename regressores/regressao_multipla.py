import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, r2_score


#PARTE 1:
# Carregar o dataset
arquivo_path = 'regressores/datasets/seno_data.csv'
data = pd.read_csv(arquivo_path)

# Defina as variáveis independentes (X) e a variável dependente (y)
X = np.array(data.iloc[:, 0:-1]) # Variaveis independentes
y = np.array(data.iloc[:,-1])    # Variavel dependente   

# Divida o conjunto de dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=42)

#PARTE 2:
# Crie um modelo de regressão Lasso com um valor de alpha (regularização) específico
model = Lasso(alpha=1.0)  # Você pode ajustar o valor de alpha conforme necessário

# Treine o modelo nos dados de treinamento
model.fit(X_train, y_train)

# Faça previsões nos dados de teste
y_pred = model.predict(X_test)

#PARTE 3:
# Avaliaçao do desempenho do modelo
# Calcule o Erro Médio Quadrático (MSE)
mse = mean_squared_error(y_test, y_pred)

# Calcule o Coeficiente de Determinação (R-squared)
r2 = r2_score(y_test, y_pred)

# Imprima as métricas
print("Erro Médio Quadrático (MSE):", mse)
print("Coeficiente de Determinação (R-squared):", r2)

# Gráfico de Dispersão: Previsões vs. Valores Reais
plt.scatter(y_test, y_pred, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--')
plt.xlabel('Valores Reais')
plt.ylabel('Previsões')
plt.title('Gráfico de Dispersão: Previsões vs. Valores Reais')
plt.show()

# Gráfico de Resíduos
residuals = y_test - y_pred
plt.scatter(y_pred, residuals, color='green')
plt.axhline(y=0, color='red', linestyle='--')
plt.xlabel('Previsões do Modelo')
plt.ylabel('Resíduos')
plt.title('Gráfico de Resíduos')
plt.show()
