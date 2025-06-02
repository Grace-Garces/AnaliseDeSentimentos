# 🎮 Análise de Sentimentos em Reviews de Jogos 📊

Este projeto realiza uma análise de sentimentos em reviews de jogos, utilizando dados de informações dos jogos e dos reviews de usuários. O objetivo principal é extrair insights sobre a percepção dos jogadores, identificar palavras-chave relevantes em diferentes contextos de sentimento e construir modelos de machine learning para prever a nota do review e a categoria de sentimento.

---

## 📑 Sumário

1.  [🌟 Visão Geral do Projeto](#-visão-geral-do-projeto)
2.  [⚙️ Estrutura do Script](#️-estrutura-do-script)
    * [2.1 Configuração do Ambiente e Dependências](#21-configuração-do-ambiente-e-dependências-)
    * [2.2 Carregamento e Pré-processamento dos Dados](#22-carregamento-e-pré-processamento-dos-dados-)
    * [2.3 Análise de Sentimento](#23-análise-de-sentimento-)
    * [2.4 Unificação dos Dados e Engenharia de Atributos](#24-unificação-dos-dados-e-engenharia-de-atributos-)
    * [2.5 Visualização de Dados (EDA)](#25-visualização-de-dados-eda--)
    * [2.6 Análise de Palavras-Chave Frequentes](#26-análise-de-palavras-chave-frequentes-)
    * [2.7 Análise de Correlações Adicionais](#27-análise-de-correlações-adicionais-)
    * [2.8 Construção de Modelos de Machine Learning](#28-construção-de-modelos-de-machine-learning-)
3.  [🚀 Como Utilizar](#-como-utilizar)
    * [3.1 Pré-requisitos](#31-pré-requisitos-)
    * [3.2 Arquivos de Dados](#32-arquivos-de-dados-)
    * [3.3 Execução](#33-execução-)
4.  [📄 Saídas do Projeto](#-saídas-do-projeto)

---

## 🌟 Visão Geral do Projeto

O script Python analisa dados de reviews de jogos para:
* Determinar o sentimento (positivo, negativo, neutro).
* Visualizar tendências e distribuições nos dados.
* Identificar termos frequentemente associados a reviews positivos e negativos.
* Construir modelos preditivos para notas de review e categorias de sentimento.

---

## ⚙️ Estrutura do Script

O script é dividido nas seguintes seções principais:

### 2.1 Configuração do Ambiente e Dependências 🛠️

* **Bibliotecas Importadas:**
    * `pandas` e `numpy` para manipulação de dados.
    * `nltk` e `vaderSentiment` para Processamento de Linguagem Natural (PLN) e análise de sentimento.
    * `matplotlib` e `seaborn` para visualização de dados.
    * `scikit-learn` para tarefas de Machine Learning.
* **Estilo dos Gráficos:** Define o estilo `seaborn-v0_8-whitegrid` e a paleta de cores `viridis`.
* **Recursos NLTK:** Verifica e baixa automaticamente os seguintes recursos do NLTK, se ausentes:
    * `vader_lexicon`
    * `punkt` (tokenizador)
    * `stopwords` (lista de palavras de parada)
* **Stopwords:** Define um conjunto de *stopwords* em inglês para remoção durante o pré-processamento de texto.

### 2.2 Carregamento e Pré-processamento dos Dados 🧹

* **Carregamento:**
    * Informações dos jogos: `games_info.csv`
    * Reviews (Parte 1): `games_review-part1.csv`
    * Reviews (Parte 2): `games_review-part2.csv`
    * Os arquivos de reviews são combinados em um único DataFrame.
* **Inspeção Inicial:** Exibe:
    * Primeiras linhas (`.head()`)
    * Dimensões (`.shape`)
    * Tipos de dados (`.info()`)
* **Limpeza e Padronização:**
    * Renomeia colunas:
        * `App Id` → `game_id`
        * `App Name` → `game_name`
    * Converte `review_date` para o formato `datetime`.
    * Remove reviews com `review_text` ausente.
    * Converte `review_text` para string.
    * Remove reviews duplicados (baseado em `game_id` e `review_text`).

### 2.3 Análise de Sentimento 😄😐😠

O sentimento é determinado por duas abordagens:

1.  **Baseado na Nota (`review_score`):**
    * Uma função `categorize_sentiment_from_score` classifica os reviews:
        * **Negativo:** Nota ≤ 2
        * **Neutro:** Nota = 3
        * **Positivo:** Nota ≥ 4
    * Calcula e exibe a distribuição percentual dessas categorias.

2.  **Com VADER (Valence Aware Dictionary and sEntiment Reasoner):**
    * Utiliza `SentimentIntensityAnalyzer` do VADER.
    * Calcula o `compound_score` para cada `review_text`.
    * Categoriza com base no `compound_score`:
        * **Positivo:** Score ≥ 0.05
        * **Negativo:** Score ≤ -0.05
        * **Neutro:** Score entre -0.05 e 0.05
    * Calcula e exibe a distribuição percentual dessas categorias.

### 2.4 Unificação dos Dados e Engenharia de Atributos 🔗

* **Merge:** Junta o DataFrame de reviews (com sentimento) com `df_games_info` usando `game_id`.
* **Métricas Agregadas por Jogo:**
    * `total_reviews`: Contagem total de reviews.
    * `average_review_score`: Média da nota do review.
    * `average_vader_compound`: Média do score composto VADER.
    * `overall_game_score`: Pontuação geral do jogo (`score`).
    * `total_ratings_count`: Contagem total de avaliações.
    * `total_downloads`: Número total de downloads.
* As métricas são ordenadas pelo `total_reviews` (decrescente) e as top 10 são exibidas.

### 2.5 Visualização de Dados (EDA) 📊📈

São gerados e salvos os seguintes gráficos:

* `distribuicao_review_score.png`: Distribuição das Notas dos Reviews.
* `distribuicao_sentimento_vader.png`: Distribuição das Categorias de Sentimento VADER.
* `top_jogos_nota_media.png`: Top 10 Jogos por Nota Média de Review (com filtro de reviews mínimos).
* `top_jogos_sentimento_vader.png`: Top 10 Jogos por Sentimento Médio VADER (com filtro de reviews mínimos).
* `correlacao_nota_sentimento_vader.png`: Relação entre Nota Média e Sentimento VADER.

### 2.6 Análise de Palavras-Chave Frequentes 📝

* **Pré-processamento de Texto para Keywords:**
    * Função `preprocess_text_for_keywords`:
        * Remove caracteres não alfabéticos.
        * Converte para minúsculas.
        * Tokeniza o texto.
        * Remove stopwords e palavras curtas (<= 2 caracteres).
* **Identificação:**
    * Extrai as palavras mais frequentes de reviews positivos e negativos (segundo VADER).
    * Exibe as Top 20 palavras para cada categoria.
* **Visualização:**
    * `top_positive_words.png`: Gráfico de barras das palavras mais frequentes em reviews positivos.
    * `top_negative_words.png`: Gráfico de barras das palavras mais frequentes em reviews negativos.

### 2.7 Análise de Correlações Adicionais 📉

* **Matriz de Correlação Numérica:**
    * Calcula a correlação entre: `review_score`, `vader_compound_score`, `score`, `ratings_count`, `downloads`, `helpful_count`.
    * Visualizada como um heatmap: `correlation_matrix.png`.
* **Sentimento por Classificação de Conteúdo:**
    * Analisa o sentimento médio VADER por `content_rating`.
    * Visualizado como um gráfico de barras: `sentiment_by_content_rating.png`.

### 2.8 Construção de Modelos de Machine Learning 🤖🧠

O script tenta construir dois modelos:

* **Preparação para ML:**
    * Cria cópia do DataFrame e remove NaNs em colunas essenciais.
    * Define features (`X`) e alvos (`y`).
    * **Divisão dos Dados:** `train_test_split` (tentativa de estratificação para classificação).
    * **Pré-processador (`ColumnTransformer`):**
        * `review_text`: `TfidfVectorizer` (stopwords, `max_features=5000`, `min_df=5`).
        * `helpful_count`: `StandardScaler`.

* **1. Modelo de Regressão (Prever Nota do Review):**
    * **Objetivo:** Prever `review_score` (1-5).
    * **Modelo:** `RandomForestRegressor` em um `Pipeline`.
    * **Avaliação:** MAE, R² Score, Acurácia (com previsões arredondadas e clipadas).
    * **Visualização:** `regressao_previsoes.png` (previsto vs. real).

* **2. Modelo de Classificação (Prever Sentimento VADER):**
    * **Objetivo:** Prever `vader_sentiment_category`.
    * **Modelo:** `LogisticRegression` em um `Pipeline`.
    * **Avaliação:** Relatório de Classificação (precisão, recall, F1-score).
    * **Visualização:** `matriz_confusao_sentimento.png` (matriz de confusão).

* **Tratamento de Erros:** Verifica dados insuficientes antes de treinar.

---

## 🚀 Como Utilizar

### 3.1 Pré-requisitos 📋

* Python 3.x
* Bibliotecas: `pandas`, `nltk`, `vaderSentiment`, `matplotlib`, `seaborn`, `scikit-learn`.
    ```bash
    pip install pandas nltk vaderSentiment matplotlib seaborn scikit-learn
    ```

### 3.2 Arquivos de Dados 📂

Certifique-se de que os seguintes arquivos CSV estão no mesmo diretório que o script Python:
* `games_info.csv`
* `games_review-part1.csv`
* `games_review-part2.csv`

(Ou ajuste os caminhos no código se estiverem em locais diferentes).

### 3.3 Execução ▶️

Execute o script Python pelo terminal:
```bash
python AnaliseSentimentos.py
