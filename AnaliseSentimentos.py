import pandas as pd
import nltk
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter
import re

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression 
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import classification_report, mean_absolute_error, r2_score, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("viridis")

try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError: 
    print("Recurso 'vader_lexicon' não encontrado. Baixando...")
    nltk.download('vader_lexicon')
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("Recurso 'punkt' não encontrado. Baixando...")
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    print("Recurso 'stopwords' não encontrado. Baixando...")
    nltk.download('stopwords')


stop_words_en = set(stopwords.words('english'))

print("Carregando os dados...")
try:

    df_games_info = pd.read_csv('games_info.csv')
    df_reviews_part1 = pd.read_csv('games_review-part1.csv')
    df_reviews_part2 = pd.read_csv('games_review-part2.csv')

    df_reviews = pd.concat([df_reviews_part1, df_reviews_part2], ignore_index=True)
    print("Dados carregados e reviews combinados com sucesso!")
except FileNotFoundError as e:
    print(f"Erro: Arquivo não encontrado. Certifique-se que os arquivos CSV estão no diretório correto.")
    print(e)
    exit()

print("\n--- Informações Iniciais ---")
print("\nGames Info (Primeiras Linhas):")
print(df_games_info.head())
print(f"\nDimensões df_games_info: {df_games_info.shape}")
print("\nTipos de dados df_games_info:")
df_games_info.info()

print("\nGames Reviews (Primeiras Linhas):")
print(df_reviews.head())
print(f"\nDimensões df_reviews: {df_reviews.shape}")
print("\nTipos de dados df_reviews:")
df_reviews.info()

if 'App Id' in df_games_info.columns and 'game_id' not in df_games_info.columns:
    df_games_info.rename(columns={'App Id': 'game_id'}, inplace=True)
if 'App Name' in df_games_info.columns and 'game_name' not in df_games_info.columns:
    df_games_info.rename(columns={'App Name': 'game_name'}, inplace=True)

if 'review_date' in df_reviews.columns:
    df_reviews['review_date'] = pd.to_datetime(df_reviews['review_date'], errors='coerce')

df_reviews.dropna(subset=['review_text'], inplace=True)
df_reviews['review_text'] = df_reviews['review_text'].astype(str)
df_reviews.drop_duplicates(subset=['game_id', 'review_text'], inplace=True)

print("\nIniciando análise de sentimento...")

def categorize_sentiment_from_score(score):
    if score is None or np.isnan(score):
        return 'Indefinido'
    if score <= 2:
        return 'Negativo'
    elif score == 3:
        return 'Neutro'
    elif score >= 4:
        return 'Positivo'
    return 'Indefinido'

if 'review_score' in df_reviews.columns:
    df_reviews['sentiment_by_score'] = df_reviews['review_score'].apply(categorize_sentiment_from_score)
    print("\nSentimento por review_score (distribuição %):")
    print(df_reviews['sentiment_by_score'].value_counts(normalize=True) * 100)

analyzer = SentimentIntensityAnalyzer()

def get_vader_compound_score(text):
    return analyzer.polarity_scores(text)['compound']

print("Aplicando VADER para análise de sentimento (pode levar um tempo)...")
df_reviews['vader_compound_score'] = df_reviews['review_text'].apply(get_vader_compound_score)
df_reviews['vader_sentiment_category'] = df_reviews['vader_compound_score'].apply(
    lambda score: 'Positivo' if score >= 0.05 else ('Negativo' if score <= -0.05 else 'Neutro')
)

print("\nSentimento por VADER (distribuição %):")
print(df_reviews['vader_sentiment_category'].value_counts(normalize=True) * 100)
print("Análise de sentimento concluída.")

print("\nJuntando dados de informações dos jogos com reviews e sentimento...")

if 'game_id' not in df_games_info.columns:
    print("ERRO: Coluna 'game_id' não encontrada em df_games_info. Verifique os nomes das colunas.")
    exit()
if 'game_id' not in df_reviews.columns:
    print("ERRO: Coluna 'game_id' não encontrada em df_reviews. Verifique os nomes das colunas.")
    exit()

df_merged = pd.merge(df_reviews, df_games_info, on='game_id', how='left')
print("Merge concluído.")
print("\nDados Unificados (Primeiras Linhas):")
print(df_merged.head())
df_merged.info()

print("\nCalculando métricas de desempenho por jogo...")

grouping_cols = ['game_id']
if 'game_name' in df_merged.columns:
    grouping_cols.append('game_name')
else:
    if 'game_name' in df_games_info.columns and 'game_id' in df_games_info.columns:
        game_name_map = df_games_info.set_index('game_id')['game_name'].to_dict()
        df_merged['game_name'] = df_merged['game_id'].map(game_name_map)
        if 'game_name' in df_merged.columns and df_merged['game_name'].notna().any():
            grouping_cols.append('game_name')


final_grouping_cols = [col for col in grouping_cols if col in df_merged.columns and df_merged[col].notna().any()]
if not final_grouping_cols:
    final_grouping_cols = ['game_id']
    if 'game_id' not in df_merged.columns:
         print("ERRO CRÍTICO: Coluna 'game_id' ausente em df_merged antes do groupby.")
         exit()


game_performance_metrics = df_merged.groupby(final_grouping_cols).agg(
    total_reviews=('review_text', 'count'),
    average_review_score=('review_score', 'mean'),
    average_vader_compound=('vader_compound_score', 'mean'),
    overall_game_score=('score', 'first'),
    total_ratings_count=('ratings_count', 'first'),
    total_downloads=('downloads', 'first')
).sort_values(by='total_reviews', ascending=False).reset_index()

print("\nMétricas de Desempenho por Jogo (Top 10 por Total de Reviews):")

print(game_performance_metrics.head(10))

print("\nGerando visualizações...")

plt.figure(figsize=(10, 6))
sns.countplot(data=df_reviews, x='review_score', order=df_reviews['review_score'].value_counts().index, palette="viridis")
plt.title('Distribuição de Notas dos Reviews (Review Score)')
plt.xlabel('Nota do Review')
plt.ylabel('Contagem')
plt.tight_layout()
plt.savefig('distribuicao_review_score.png')
plt.show()

plt.figure(figsize=(10, 6))
sns.countplot(data=df_reviews, x='vader_sentiment_category', order=['Positivo', 'Neutro', 'Negativo'], palette="viridis")
plt.title('Distribuição de Sentimento (Categorias VADER)')
plt.xlabel('Categoria de Sentimento VADER')
plt.ylabel('Contagem')
plt.tight_layout()
plt.savefig('distribuicao_sentimento_vader.png')
plt.show()

min_reviews_threshold = 50
top_games_by_score = game_performance_metrics[game_performance_metrics['total_reviews'] >= min_reviews_threshold].sort_values(
    by='average_review_score', ascending=False
).head(10)

if not top_games_by_score.empty and 'game_name' in top_games_by_score.columns:
    plt.figure(figsize=(12, 8))
    sns.barplot(data=top_games_by_score, x='average_review_score', y='game_name', palette='Spectral')
    plt.title(f'Top 10 Jogos por Nota Média de Review (Mín. {min_reviews_threshold} Reviews)')
    plt.xlabel('Nota Média do Review')
    plt.ylabel('Jogo')
    plt.xlim(0, 5)
    plt.tight_layout()
    plt.savefig('top_jogos_nota_media.png')
    plt.show()
elif not top_games_by_score.empty:
    print("Top jogos por nota média calculado, mas 'game_name' não disponível para plotagem do nome.")
    # Removed .to_markdown()
    print(top_games_by_score[['game_id', 'average_review_score', 'total_reviews']])
else:
    print(f"Nenhum jogo encontrado com pelo menos {min_reviews_threshold} reviews para o gráfico de Top Jogos por Nota.")

top_games_by_vader = game_performance_metrics[game_performance_metrics['total_reviews'] >= min_reviews_threshold].sort_values(
    by='average_vader_compound', ascending=False
).head(10)

if not top_games_by_vader.empty and 'game_name' in top_games_by_vader.columns:
    plt.figure(figsize=(12, 8))
    sns.barplot(data=top_games_by_vader, x='average_vader_compound', y='game_name', palette='coolwarm')
    plt.title(f'Top 10 Jogos por Sentimento Médio VADER (Mín. {min_reviews_threshold} Reviews)')
    plt.xlabel('Sentimento Médio VADER (Compound Score)')
    plt.ylabel('Jogo')
    plt.xlim(-1, 1)
    plt.tight_layout()
    plt.savefig('top_jogos_sentimento_vader.png')
    plt.show()
elif not top_games_by_vader.empty:
    print("Top jogos por sentimento VADER calculado, mas 'game_name' não disponível para plotagem do nome.")
    print(top_games_by_vader[['game_id', 'average_vader_compound', 'total_reviews']])
else:
    print(f"Nenhum jogo encontrado com pelo menos {min_reviews_threshold} reviews para o gráfico de Top Jogos por Sentimento VADER.")

if 'average_review_score' in game_performance_metrics.columns and 'average_vader_compound' in game_performance_metrics.columns:
    plt.figure(figsize=(10, 7))
    sns.scatterplot(
        data=game_performance_metrics[game_performance_metrics['total_reviews'] >= min_reviews_threshold],
        x='average_review_score',
        y='average_vader_compound',
        size='total_reviews',
        hue='total_reviews', 
        sizes=(50, 500),
        palette="viridis", 
        alpha=0.7
    )
    plt.title(f'Nota Média vs. Sentimento VADER (Mín. {min_reviews_threshold} Reviews)')
    plt.xlabel('Nota Média do Review (1-5)')
    plt.ylabel('Sentimento Médio VADER (-1 a 1)')
    plt.axhline(0, color='grey', linestyle='--')
    plt.axvline(3, color='grey', linestyle='--') 
    plt.legend(title='Total de Reviews', loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.savefig('correlacao_nota_sentimento_vader.png')
    plt.show()
else:
    print("Não foi possível gerar o gráfico de correlação: colunas 'average_review_score' ou 'average_vader_compound' ausentes.")



print("\n--- Análise de Palavras-Chave Frequentes ---")

def preprocess_text_for_keywords(text):

    text = re.sub(r'[^a-zA-Z\s]', '', text).lower() 
    tokens = word_tokenize(text) 
    tokens = [word for word in tokens if word not in stop_words_en and len(word) > 2]
    return tokens


if 'review_text' in df_merged.columns:
    df_merged['processed_review_text'] = df_merged['review_text'].apply(preprocess_text_for_keywords)

    positive_reviews = df_merged[df_merged['vader_sentiment_category'] == 'Positivo']['processed_review_text']
    negative_reviews = df_merged[df_merged['vader_sentiment_category'] == 'Negativo']['processed_review_text']

    all_positive_words = [word for sublist in positive_reviews for word in sublist]
    positive_word_freq = Counter(all_positive_words)
    print("\nTop 20 Palavras Mais Frequentes em Reviews Positivos:")
    for word, freq in positive_word_freq.most_common(20):
        print(f"- {word}: {freq}")

    all_negative_words = [word for sublist in negative_reviews for word in sublist]
    negative_word_freq = Counter(all_negative_words)
    print("\nTop 20 Palavras Mais Frequentes em Reviews Negativos:")
    for word, freq in negative_word_freq.most_common(20):
        print(f"- {word}: {freq}")

    def plot_word_frequencies(word_freq, title, filename):
        if not word_freq:
            print(f"Não há dados de frequência de palavras para '{title}'. Pulando plotagem.")
            return
        df_freq = pd.DataFrame(word_freq.most_common(20), columns=['Palavra', 'Frequência'])
        if df_freq.empty:
            print(f"DataFrame de frequência de palavras para '{title}' está vazio. Pulando plotagem.")
            return
        plt.figure(figsize=(12, 8))
        sns.barplot(x='Frequência', y='Palavra', data=df_freq, palette='magma')
        plt.title(title)
        plt.xlabel('Frequência')
        plt.ylabel('Palavra')
        plt.tight_layout()
        plt.savefig(filename)
        plt.show()

    plot_word_frequencies(positive_word_freq, 'Top 20 Palavras em Reviews Positivos', 'top_positive_words.png')
    plot_word_frequencies(negative_word_freq, 'Top 20 Palavras em Reviews Negativos', 'top_negative_words.png')
else:
    print("Coluna 'review_text' não encontrada em df_merged. Pulando análise de palavras-chave.")


print("\n--- Análise de Correlações Mais Profundas ---")

numerical_cols = ['review_score', 'vader_compound_score', 'score', 'ratings_count', 'downloads', 'helpful_count']
existing_numerical_cols = [col for col in numerical_cols if col in df_merged.columns and pd.api.types.is_numeric_dtype(df_merged[col])]


if len(existing_numerical_cols) > 1:
    correlation_matrix = df_merged[existing_numerical_cols].corr()
    print("\nMatriz de Correlação entre Variáveis Numéricas:")
    print(correlation_matrix)

    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
    plt.title('Matriz de Correlação de Variáveis Numéricas')
    plt.tight_layout()
    plt.savefig('correlation_matrix.png')
    plt.show()
else:
    print(f"Não há colunas numéricas suficientes ({len(existing_numerical_cols)} encontradas: {existing_numerical_cols}) para calcular a matriz de correlação.")

if 'content_rating' in df_merged.columns and 'vader_compound_score' in df_merged.columns:
    print("\nSentimento Médio VADER por Classificação de Conteúdo:")
    sentiment_by_content_rating = df_merged.dropna(subset=['content_rating']).groupby('content_rating')['vader_compound_score'].mean().sort_values(ascending=False)
    print(sentiment_by_content_rating)

    if not sentiment_by_content_rating.empty:
        plt.figure(figsize=(10, 6))
        sns.barplot(x=sentiment_by_content_rating.index, y=sentiment_by_content_rating.values, palette='viridis')
        plt.title('Sentimento Médio VADER por Classificação de Conteúdo')
        plt.xlabel('Classificação de Conteúdo')
        plt.ylabel('Sentimento Médio VADER')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig('sentiment_by_content_rating.png')
        plt.show()
    else:
        print("Não há dados de sentimento por classificação de conteúdo para plotar.")


print("\n--- Construindo Modelos de Machine Learning ---")

df_ml = df_merged.copy()


ml_required_cols = ['review_text', 'review_score', 'vader_sentiment_category', 'helpful_count']
missing_cols_for_ml = [col for col in ml_required_cols if col not in df_ml.columns]

if missing_cols_for_ml:
    print(f"ERRO: Colunas necessárias para ML estão faltando: {missing_cols_for_ml}. Saindo da seção de ML.")
else:
    df_ml.dropna(subset=ml_required_cols, inplace=True)

    if df_ml.empty:
        print("DataFrame para ML está vazio após remover NaNs. Não é possível treinar modelos.")
    else:
        df_ml['review_score'] = df_ml['review_score'].astype(int)

        X_reg = df_ml[['review_text', 'helpful_count']]
        y_reg = df_ml['review_score']

        X_clf = df_ml[['review_text', 'helpful_count']]
        y_clf = df_ml['vader_sentiment_category']


        if len(df_ml) < 2 or len(y_reg.unique()) < 2 and len(y_clf.unique()) < 2 : 
             print("Dados insuficientes para treinar os modelos após o pré-processamento. Pulando a etapa de ML.")
        else:
            try:
                X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)
  
                if len(y_clf.value_counts()[y_clf.value_counts() < 2]) > 0 and len(y_clf.unique()) > 1 :
                    print("Aviso: Algumas classes em y_clf têm menos de 2 amostras. Usando stratify pode falhar. Tentando sem stratify se necessário.")
                    try:
                         X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(X_clf, y_clf, test_size=0.2, random_state=42, stratify=y_clf)
                    except ValueError:
                         print("Falha ao usar stratify. Tentando sem stratify.")
                         X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(X_clf, y_clf, test_size=0.2, random_state=42)
                elif len(y_clf.unique()) <= 1:
                     print("Apenas uma classe em y_clf. Não é possível treinar o classificador ou usar stratify. Pulando modelo de classificação.")
                     model_clf = None 
                else:
                    X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(X_clf, y_clf, test_size=0.2, random_state=42, stratify=y_clf)


                preprocessor = ColumnTransformer(
                    transformers=[
                        ('text', TfidfVectorizer(stop_words=list(stop_words_en), max_features=5000, min_df=5), 'review_text'), # Added min_df
                        ('num', StandardScaler(), ['helpful_count'])
                    ],
                    remainder='passthrough'
                )

                print("\n--- Modelo de Regressão: Prever Nota do Review ---")

                if y_train_reg.nunique() > 1 and not X_train_reg.empty :
                    model_reg = Pipeline(steps=[('preprocessor', preprocessor),
                                                ('regressor', RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1, min_samples_split=10, min_samples_leaf=5))]) # Added min_samples

                    model_reg.fit(X_train_reg, y_train_reg)
                    y_pred_reg = model_reg.predict(X_test_reg)

                    y_pred_reg_rounded = np.round(y_pred_reg)
                    y_pred_reg_clipped = np.clip(y_pred_reg_rounded, 1, 5) 

                    mae_reg = mean_absolute_error(y_test_reg, y_pred_reg_clipped)
                    r2_reg = r2_score(y_test_reg, y_pred_reg_clipped)
                    accuracy_reg = accuracy_score(y_test_reg, y_pred_reg_clipped)

                    print(f"MAE (Mean Absolute Error) do Modelo de Regressão: {mae_reg:.4f}")
                    print(f"R2 Score do Modelo de Regressão: {r2_reg:.4f}")
                    print(f"Acurácia (previsão exata da nota) do Modelo de Regressão: {accuracy_reg:.4f}")

                    plt.figure(figsize=(10, 7))
                    sns.regplot(x=y_test_reg, y=y_pred_reg_clipped, scatter_kws={'alpha':0.3}, line_kws={'color':'red'})
                    plt.title('Previsões do Modelo de Regressão vs. Valores Reais (Notas Arredondadas e Cortadas)')
                    plt.xlabel('Nota Real do Review')
                    plt.ylabel('Nota Prevista do Review')
                    plt.grid(True)
                    plt.tight_layout()
                    plt.savefig('regressao_previsoes.png')
                    plt.show()
                else:
                    print("Dados insuficientes ou sem variação para treinar o modelo de regressão.")


                if 'model_clf' in locals() and model_clf is None:
                    print("\n--- Modelo de Classificação: Prever Sentimento VADER ---")
                    print("Pulando modelo de classificação devido a apenas uma classe presente nos dados de treino.")
                elif y_train_clf.nunique() > 1 and not X_train_clf.empty:
                    print("\n--- Modelo de Classificação: Prever Sentimento VADER ---")
                    model_clf_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                                ('classifier', LogisticRegression(random_state=42, solver='liblinear', multi_class='auto', C=1.0))]) # Added C

                    model_clf_pipeline.fit(X_train_clf, y_train_clf)
                    y_pred_clf = model_clf_pipeline.predict(X_test_clf)

                    print("\nRelatório de Classificação para Sentimento VADER:")
                   
                    clf_labels = sorted(list(set(y_test_clf) | set(y_pred_clf)))
                    if not clf_labels: 
                        clf_labels = ['Positivo', 'Neutro', 'Negativo']

                    print(classification_report(y_test_clf, y_pred_clf, labels=clf_labels, zero_division=0))

                    cm = confusion_matrix(y_test_clf, y_pred_clf, labels=clf_labels)
                    plt.figure(figsize=(8, 6))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=clf_labels, yticklabels=clf_labels)
                    plt.title('Matriz de Confusão para Previsão de Sentimento')
                    plt.xlabel('Sentimento Previsto')
                    plt.ylabel('Sentimento Real')
                    plt.tight_layout()
                    plt.savefig('matriz_confusao_sentimento.png')
                    plt.show()
                else:
                    print("Dados insuficientes ou sem variação para treinar o modelo de classificação.")
            except ValueError as ve:
                 print(f"Erro durante o split ou treinamento dos modelos de ML: {ve}")
                 print("Pode ser devido a dados insuficientes ou classes desbalanceadas. Pulando seção de ML.")


print("\nAnálise, visualizações e modelos de ML concluídos (se os dados permitiram). Verifique os arquivos .png salvos no diretório.")
