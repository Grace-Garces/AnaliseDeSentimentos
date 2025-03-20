import json
import pandas as pd
import re
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Carregar JSON
caminho_json = "dados.json"
with open(caminho_json, "r", encoding="utf-8") as f:
    dados = json.load(f)

df = pd.DataFrame(dados)

# Criar coluna "Sentimento"
def definir_sentimento(nota):
    return "Positivo" if nota >= 4 else "Negativo" if nota <= 2 else "Neutro"

df["Sentimento"] = df["Ratings"].apply(definir_sentimento)

# Lista de stopwords customizadas
stopwords_personalizadas = {"the", "a", "an", "is", "are", "i", "you", "we", "they", "this", "that", "and", "or", "but"} | {"chatgpt", "app", "software", "ai", "bot", "gpt"}
palavras_positivas = {"good", "great", "excellent"}

# Função para limpar texto
def limpar_texto(texto, remover_palavras_positivas=False):
    if not isinstance(texto, str):  # Verifica se é string, senão define como vazio
        texto = ""
    texto = texto.lower()
    texto = re.sub(r'\bchat\s?gpt\b', '', texto)  # Remove 'chat gpt' do texto

    palavras = texto.split()  # Tokeniza o texto manualmente usando o split

    # Filtra as palavras
    palavras_filtradas = [p for p in palavras if p.isalpha() and p not in stopwords_personalizadas]

    # Remove as palavras positivas, se necessário
    if remover_palavras_positivas:
        palavras_filtradas = [word for word in palavras_filtradas if word not in palavras_positivas]

    return " ".join(palavras_filtradas)

# Aplicar limpeza
df["Review_Limpo"] = df.apply(lambda row: limpar_texto(row["Review"], remover_palavras_positivas=row["Sentimento"] == "Negativo"), axis=1)

# Criar textos para WordCloud
texto_positivo = " ".join(df[df["Sentimento"] == "Positivo"]["Review_Limpo"].dropna())
texto_negativo = " ".join(df[df["Sentimento"] == "Negativo"]["Review_Limpo"].dropna())

plt.figure(figsize=(12, 5))

# Nuvem de palavras positivas
wordcloud_pos = WordCloud(width=800, height=400, background_color="white").generate(texto_positivo)
plt.subplot(1, 2, 1)
plt.imshow(wordcloud_pos, interpolation="bilinear")
plt.axis("off")
plt.title("🔹 Palavras mais frequentes em avaliações POSITIVAS")

# Nuvem de palavras negativas
if texto_negativo:
    wordcloud_neg = WordCloud(width=800, height=400, background_color="black", colormap="Reds").generate(texto_negativo)
    plt.subplot(1, 2, 2)
    plt.imshow(wordcloud_neg, interpolation="bilinear")
    plt.axis("off")
    plt.title("🔻 Palavras mais frequentes em avaliações NEGATIVAS")

plt.show()
