# Análise Estatística do Índice Big Mac (R)

Este projeto realiza a análise estatística histórica global de paridade de poder de compra (PPP - Purchasing Power Parity) e variabilidade de câmbio utilizando o **Índice Big Mac** (publicado pela revista *The Economist*).

O projeto foi originalmente desenvolvido como uma consultoria estatística para o curso de Economia e Administração da **FEA-USP**.

---

## 🛠️ Tecnologias e Recursos Utilizados

- **Linguagem:** R
- **Formato de Documentação:** RMarkdown (`.Rmd`)
- **Bibliotecas e Funções Nativas:** Manipulação de vetores, manipulação de dataframes, estatística básica (`mean`, `var`), agregação estruturada de subconjuntos (`aggregate`, `split`, `sapply`, `lapply`), e plotagem gráfica base (`barplot`).
- **Base de Dados:** `epData.csv` contendo registros históricos globais de preços locais e taxas de câmbio.

---

## 📊 Arquitetura de Análise

1. **Processamento de Dados:** Funções personalizadas para tratamento de datas, filtragem temporal e correspondência de preços internacionais indexados pelo dólar americano.
2. **Cálculo Estatístico Customizado:** Algoritmos em R para cálculo de médias históricas ponderadas e variância local sem o uso de dependências externas complexas.
3. **Análise Macro-Regional:** Agrupamento de países por continentes e hemisférios para comparação do poder de compra médio global.
4. **Visualização de Dados:** Geração de gráficos de barras para representação de custos médios e variância histórica de preços locais frente ao Dólar.

---

## 🚀 Como Visualizar e Publicar

### 1. Visualizar Localmente (RStudio)
1. Abra o arquivo `indice_big_mac.Rmd` no **RStudio**.
2. Garanta que o arquivo de dados `epData.csv` esteja na mesma pasta.
3. Clique no botão **Knit** (ou use o atalho `Ctrl + Shift + K`) para gerar um relatório em formato **HTML** autônomo, limpo e com gráficos interativos de forma local.

### 2. Publicar Online (RPubs - Gratuito)
1. Após gerar o HTML com o **Knit** no RStudio, clique no botão **Publish** (localizado no canto superior direito do painel de visualização HTML do RStudio).
2. Selecione a plataforma **RPubs**.
3. Crie uma conta gratuita (ou faça login) e confirme a publicação.
4. Você terá uma URL permanente e profissional para compartilhar em seu currículo e no LinkedIn!
