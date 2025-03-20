# Análise de Sentimentos de Avaliações do ChatGPT

##  Sobre o Projeto
Este projeto tem como objetivo realizar uma **análise de sentimentos** das avaliações do ChatGPT, explorando a distribuição das notas atribuídas pelos usuários e identificando padrões em sentimentos positivos e negativos. Utilizamos **Python, Pandas, Matplotlib, Seaborn e WordCloud** para carregar, tratar, visualizar e interpretar os dados.

##  Tecnologias Utilizadas
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

##  Como Executar
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

##  Análise dos Dados
###  Tratamento de Dados
O script carrega os dados do arquivo JSON, os converte em um **DataFrame Pandas** e realiza as seguintes operações:
- **Verificação de valores nulos** e duplicados.
- **Distribuição das notas** atribuídas pelos usuários.
- **Exibição de exemplos** de avaliações positivas e negativas.

###  Visualização dos Sentimentos
O projeto gera:
- Um **gráfico de barras** mostrando a distribuição das notas.
- **Nuvens de palavras** destacando os termos mais comuns em avaliações positivas e negativas.
![image](https://github.com/user-attachments/assets/869a7cb4-e433-4d97-9854-489b8c2037bb)

