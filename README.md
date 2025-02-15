# Análise de Sentimentos de Avaliações do ChatGPT

## 📌 Sobre o Projeto
Este projeto tem como objetivo realizar uma **análise de sentimentos** das avaliações do ChatGPT, explorando a distribuição das notas atribuídas pelos usuários e identificando padrões em sentimentos positivos e negativos. Utilizamos **Python, Pandas, Matplotlib, Seaborn e WordCloud** para carregar, tratar, visualizar e interpretar os dados.

## 🔧 Tecnologias Utilizadas
- **Python** → Linguagem principal para manipulação e análise de dados.
- **Pandas** → Para processamento e análise de dados.
- **Matplotlib e Seaborn** → Para visualização gráfica dos dados.
- **WordCloud** → Para gerar nuvens de palavras baseadas nos sentimentos expressos nas avaliações.

## 📂 Estrutura do Projeto
```
AnaliseDeSentimentos/
│-- Data/
│   ├── ChatGPT_Reviews_Sentimentos.json  # Arquivo com as avaliações
│-- Scripts/
│   ├── analise_sentimentos.py  # Script principal de análise
│   ├── wordcloud_sentimentos.py  # Geração de nuvem de palavras
│-- README.md  # Documentação do projeto
```

## 🚀 Como Executar
1. Clone o repositório:
```bash
git clone https://github.com/seu-usuario/AnaliseDeSentimentos.git
```
2. Instale as dependências necessárias:
```bash
pip install pandas matplotlib seaborn wordcloud
```
3. Execute a análise inicial:
```bash
python Scripts/analise_sentimentos.py
```
4. Gere a nuvem de palavras:
```bash
python Scripts/wordcloud_sentimentos.py
```

## 📊 Análise dos Dados
### 🛠️ Tratamento de Dados
O script carrega os dados do arquivo JSON, os converte em um **DataFrame Pandas** e realiza as seguintes operações:
- **Verificação de valores nulos** e duplicados.
- **Distribuição das notas** atribuídas pelos usuários.
- **Exibição de exemplos** de avaliações positivas e negativas.

### 📊 Visualização dos Sentimentos
O projeto gera:
- Um **gráfico de barras** mostrando a distribuição das notas.
- **Nuvens de palavras** destacando os termos mais comuns em avaliações positivas e negativas.

### 🔍 Filtragem de Sentimentos
O script também busca padrões de expressões nas avaliações negativas:
- Frases contendo **"not good" ou "no good"**.
- Avaliações negativas que mencionam a palavra **"good"**, indicando ironia ou contexto atenuado.

## 🖼️ Exemplos de Saídas
### 📊 Gráfico de Distribuição das Notas
![Distribuição das Notas](https://github.com/seu-usuario/AnaliseDeSentimentos/blob/main/assets/distribuicao_notas.png)

### ☁️ Nuvem de Palavras
**Avaliações Positivas:**
![Nuvem Positiva](https://github.com/seu-usuario/AnaliseDeSentimentos/blob/main/assets/nuvem_positiva.png)

**Avaliações Negativas:**
![Nuvem Negativa](https://github.com/seu-usuario/AnaliseDeSentimentos/blob/main/assets/nuvem_negativa.png)

## 📝 Conclusão
Este projeto demonstra como a análise de sentimentos pode ser aplicada para extrair insights de avaliações de usuários, identificando padrões e visualizando tendências por meio de gráficos e nuvens de palavras. Ele pode ser expandido com modelos de **Machine Learning** para classificação automática dos sentimentos.

## 📌 Melhorias Futuras
- Implementação de **modelo de Machine Learning** para classificação automática de sentimentos.
- Melhor tratamento de linguagem natural (NLP) para **detecção de ironia e sarcasmo**.
- Expansão para outras bases de avaliações e produtos.

## 📩 Contato
📧 Email: gracebatista152@gmail.com
🔗 LinkedIn: [Linkedin](https://www.linkedin.com/in/grace-garces-103174210/)

