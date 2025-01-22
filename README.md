<h1>Nike Global Sales Data (2024)</h1>

<p>Este projeto analisa os dados de vendas da Nike no ano de 2024, com o objetivo de entender o desempenho da marca em diferentes mercados e identificar tendências de consumo. A análise inclui a exploração de dados, visualizações e insights estratégicos para apoiar decisões de negócios e estratégias de marketing</p>

Fonte: [DATASET NIKE](https://www.kaggle.com/datasets/ayushcx/nike-global-sales-data-2024)

<hr>

### Estrutura do repositório

LICENSE - Arquivo de licença do repositório<br>
README.md - Arquivo readme do repositório<br>
NIKE.ipynb - Arquivo jupyter notebook criado no google colaboratory<br>
nike_sales_2024.csv - Arquivo .csv onde contém as informações usadas para análise<br>

### Estrutura do arquivo .csv

Month: Mês da atividade de vendas<br>
Region: Região geográfica (por exemplo, Grande China, Europa, América)<br>
Main_Category: Categoria principal do produto (Calçados, Roupas, Equipamentos)<br>
Sub_Category: Subcategoria específica dentro da categoria principal<br>
Product_Line: Linha de produtos ou modelos específicos<br>
Price_Tier: Segmento de preços (Premium, Médio, Econômico)<br>
Units_Sold: Número de itens vendidos<br>
Revenue_USD: Receita total em Dólares Americanos<br>
Online_Sales_Percentage: Percentual de vendas através de plataformas online<br>
Retail_Price: Preço de varejo por unidade em Dólares Americanos<br>

### Variáveis criadas

vendas_regiao<br>
receita_categoria<br>
receita_categoria_milhares<br>
receita_subcategoria<br>
vendas_online<br>
modelo<br>
y_pred<br>
tendencia_receita<br>
receita_mensal<br>
categoria_regiao<br>

<hr>

<h1>Análise</h1>

<h2>1. Bibliotecas</h2>

```python
# Bibliotecas principais
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from google_drive_downloader import GoogleDriveDownloader as gdd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

sns.set_style("whitegrid")

# Configurações gerais
plt.rcParams['figure.figsize'] = (12, 6)
pd.options.display.float_format = '{:.2f}'.format
```

<h2>2. Carregando e visualizando a base de dados</h2>

```python
# Carregando dados do Google Drive
dadosgoogleid = '1etKSfPx4riaMlDrevvqvzhY5GKykZj2z'

gdd.download_file_from_google_drive(file_id=dadosgoogleid, dest_path = './dados_google_drive.csv',showsize = True)

data = pd.read_csv("dados_google_drive.csv", sep = ',')
```
```
# Visualizando as primeiras linhas
data.head()
```
```
# Resumo estatístico e informações gerais
data.info()
data.describe()
```
```
print(data['Month'].unique())
```

<h2>3. Análise exploratória de vendas e receitas</h2>
<h3>3.1 Vendas por regiâo</h3>

```
# Total de vendas por região
vendas_regiao = data.groupby('Region')['Units_Sold'].sum().sort_values(ascending=False)

# Gráfico de barras com valores
ax = vendas_regiao.plot(kind='bar', color='skyblue', title="Total de vendas por região")
plt.ylabel('Unidades Vendidas (Milhares de dólares)')  # Escala adicionada aqui
plt.xlabel('Região')

# Adicionando os valores nas barras
for i, valor in enumerate(vendas_regiao):
    ax.text(i, valor + 0.02 * vendas_regiao.max(), f'{valor:.0f}', ha='center', va='bottom')

plt.show()
```
![download](https://github.com/user-attachments/assets/7163f077-3f31-4b25-b171-ef0275992f2b)

<h3>3.2 Receitas por categoria</h3>

```
# Receita total por categoria
receita_categoria = data.groupby('Main_Category')['Revenue_USD'].sum().sort_values(ascending=False)

# Converte a receita para milhares
receita_categoria_milhares = receita_categoria / 1_000

# Gráfico de barras com valores
ax = receita_categoria_milhares.plot(kind='bar', color='skyblue', title="Receita total por categoria")
plt.ylabel('Receita (Milhares de dólares)')  # Unidade de medida no eixo Y
plt.xlabel('Categoria')

# Adicionando os valores nas barras
for i, valor in enumerate(receita_categoria_milhares):
    ax.text(i, valor + 0.02 * receita_categoria_milhares.max(), f'${valor:.0f}', ha='center', va='bottom')

plt.show()
```
![download](https://github.com/user-attachments/assets/c81b68b0-1cba-48d4-83d4-a41b453cf54b)

<h3>3.3 Unidades vendidas por categoria</h3>

```
unidades_vendidas = data.groupby('Main_Category')['Units_Sold'].sum()
unidades_vendidas.plot(kind='pie', autopct='%1.1f%%', title='Distribuição de unidades vendidas por categoria', figsize=(6, 6))
plt.ylabel('')
plt.show()
```
![download](https://github.com/user-attachments/assets/c3992c21-567e-4e98-a3bd-3117a298385a)

<h3>3.4 Receita por subcategoria</h3>

```
# Agrupando os dados por Sub_Category e somando a Receita (USD)
receita_subcategoria = data.groupby('Sub_Category')['Revenue_USD'].sum().sort_values(ascending=False)

# Criando o gráfico de barras
plt.figure(figsize=(10, 6))
receita_subcategoria.plot(kind='bar', color='skyblue')

# Adicionando título e rótulos aos eixos
plt.title('Receita por subcategoria', fontsize=16)
plt.xlabel('Subcategoria', fontsize=12)
plt.ylabel('Receita (Milhares de dólares)', fontsize=12)

# Exibindo os valores nas barras
for i, v in enumerate(receita_subcategoria):
    plt.text(i, v + 50000, f'${v:,.0f}', ha='center', va='bottom', fontsize=10)

plt.xticks(rotation=45, ha='right')  # Rotaciona os rótulos do eixo X
plt.tight_layout()  # Ajusta o layout para evitar sobreposição
plt.show()
```
![download](https://github.com/user-attachments/assets/189d8a86-55d4-419f-a1a8-3bfb7c3531b2)

<h3>3.5 Unidades vendidas por subcategoria</h3>

```
# Agrupando os dados por Sub_Category e somando as Unidades Vendidas
unidades_subcategoria = data.groupby('Sub_Category')['Units_Sold'].sum().sort_values(ascending=False)

# Criando o gráfico de barras
plt.figure(figsize=(10, 6))
unidades_subcategoria.plot(kind='bar', color='skyblue')

# Adicionando título e rótulos aos eixos
plt.title('Unidades vendidas por subcategoria', fontsize=16)
plt.xlabel('Subcategoria', fontsize=12)
plt.ylabel('Unidades Vendidas', fontsize=12)

# Exibindo os valores nas barras
for i, v in enumerate(unidades_subcategoria):
    plt.text(i, v + 50, f'{v:,.0f}', ha='center', va='bottom', fontsize=10)

plt.xticks(rotation=45, ha='right')  # Rotaciona os rótulos do eixo X
plt.tight_layout()  # Ajusta o layout para evitar sobreposição
plt.show()
```
![download](https://github.com/user-attachments/assets/993d7481-2149-4b08-8b19-4527f1c40cb9) 

<h3>3.6 Unidades vendidas por subcategoria</h3>

```
# Proporção de vendas online
vendas_online = data['Online_Sales_Percentage'].mean()
print(f"Percentual médio de vendas online: {vendas_online:.2f}%")
```
![{76BF5EB8-8256-4447-999E-0332F3D01113}](https://github.com/user-attachments/assets/a0069908-79dd-40eb-a3c0-0cf5d38e53ca)

<h2>4. Modelos de previsão de vendas</h2>

```
# Pré-processamento
X = data[['Units_Sold', 'Online_Sales_Percentage', 'Retail_Price']]
y = data['Revenue_USD']

# Divisão em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modelo
modelo = RandomForestRegressor(random_state=42)
modelo.fit(X_train, y_train)

# Previsões
y_pred = modelo.predict(X_test)

# Avaliação
mse = mean_squared_error(y_test, y_pred)  # Calcula o MSE
rmse = np.sqrt(mse)  # Calcula o RMSE manualmente
print(f"RMSE: {rmse:.2f}")
```
![{E7AC47D1-A276-4238-94CC-78ED97AC76C2}](https://github.com/user-attachments/assets/9a611c73-1588-4480-b412-e6721ea1e361)

<p>O pré-processamento incluiu a seleção das variáveis de entrada (Units_Sold, Online_Sales_Percentage, Retail_Price) e a variável alvo (Revenue_USD). Em seguida, os dados foram divididos em conjuntos de treinamento e teste utilizando uma proporção de 80% para treinamento e 20% para teste.

Para a modelagem, foi utilizado um RandomForestRegressor, que foi treinado com os dados de treinamento. O modelo foi avaliado utilizando o Root Mean Squared Error (RMSE), que forneceu um valor de 148,194.38, indicando a magnitude do erro das previsões do modelo.

O RMSE é uma métrica comum para avaliar a precisão dos modelos de regressão. Quanto menor o valor do RMSE, melhor o modelo se ajusta aos dados reais. Nesse caso, um RMSE de 148,194.38 significa que, em média, as previsões de receita do modelo estão com um erro de cerca de 148 mil dólares, o que pode ser considerado razoável dependendo da magnitude das receitas reais e do objetivo de precisão do modelo.

<h2>5. Identificação de tendências de mercado e análise regional</h2>
<h3>5.1 Tendências de receita ao longo do ano</h3>

```
tendencia_receita = data.groupby('Month')['Revenue_USD'].sum()
tendencia_receita.plot(marker='o', title='Tendência de receita mensal', figsize=(15, 5))
plt.ylabel('Receita (Milhares de dólares)')
plt.xlabel('Mês')
plt.show()
```
![download](https://github.com/user-attachments/assets/e3e63ae6-1543-4157-b31d-2250228e6518)


<h3>5.2 Análise temporal por região</h3>

```
# Adicionando o ano ao nome do mês
data['Month'] = data['Month'] + ' 2024'

# Convertendo para datetime
data['Month'] = pd.to_datetime(data['Month'], format='%B %Y')
```
```
# Receita mensal por região
data['Month'] = pd.to_datetime(data['Month'], format='%Y-%m')

receita_mensal = data.groupby(['Month', 'Region'])['Revenue_USD'].sum().unstack()
receita_mensal.plot(marker='o', title="Receita mensal por região")
plt.ylabel('Receita (Milhares de dólares)')
plt.xlabel('Mês')
plt.show()
```
![download](https://github.com/user-attachments/assets/06668664-29ab-45fc-b311-f2cd649d39d1)

<h3>5.3 Categorias Mais Vendidas por Região</h3>

```
# Categorias populares por região
categoria_regiao = data.groupby(['Region', 'Main_Category'])['Units_Sold'].sum().unstack()
sns.heatmap(categoria_regiao, annot=True, fmt=".0f", cmap="Blues", cbar=True)
plt.title("Categorias populares por região")
plt.ylabel('Região')
plt.xlabel('Categoria')
plt.show()
```
![download](https://github.com/user-attachments/assets/825bfdc7-8836-4449-90f1-01382b2e6d74)




