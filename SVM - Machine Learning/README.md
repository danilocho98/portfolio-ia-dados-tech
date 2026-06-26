# Classificador de Sentimentos (Machine Learning — SVM)

Este repositório contém a implementação de um modelo de Machine Learning em Python para **Classificação de Sentimentos** em textos (análise de opinião binária/polaridade de frases). 

O projeto explora a matemática e a aplicação prática de **Support Vector Machines (SVM)** para problemas de Processamento de Linguagem Natural (PLN).

---

## 🛠️ Tecnologias e Recursos Utilizados

- **Linguagem:** Python
- **Ferramenta de Desenvolvimento:** Jupyter Notebook (`.ipynb`)
- **Bibliotecas Principais:** `scikit-learn` (modelagem), `pandas` (manipulação de dados), `numpy` (cálculo de vetores) e `matplotlib` (visualização de dados).
- **Algoritmo:** Support Vector Machine (SVM) com ajuste de margem e hiperparâmetros de regularização ($C$ e funções de kernel).

---

## 📊 Arquitetura do Projeto

1. **Pré-Processamento de Texto:** Tokenização de sentenças, limpeza de caracteres especiais e estruturação de dados brutos.
2. **Representação de Texto (Vetorização):** Transformação de texto em vetores numéricos para treinamento de modelo de dados.
3. **Treinamento e Modelagem:**
   - Implementação de um classificador SVM.
   - Ajuste fino dos hiperparâmetros de penalidade de erro (Soft Margin vs. Hard Margin) para evitar overfitting.
4. **Avaliação do Modelo:**
   - Teste de predição sobre novas frases de controle (exemplos positivos e negativos).
   - Otimização que levou a uma **acurácia de ~80.1%** no conjunto de validação.

---

## 🚀 Como Executar e Testar

### 1. Executar Online no Google Colab (Recomendado)
Você pode abrir e rodar este notebook diretamente na nuvem (sem precisar instalar nada na sua máquina) clicando no botão abaixo:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dcho98/portfolio-ia-dados/blob/main/SVM%20-%20Machine%20Learning/Danilo%20Cho%20-%20EP1-SVM-Classific_Sentimentos-2024-alunos.ipynb)

*(Nota: Lembre-se de ajustar a URL acima com o caminho exato do seu repositório quando subir o arquivo para o seu GitHub).*

### 2. Executar Localmente
1. Certifique-se de ter o Python e o Jupyter instalados em sua máquina.
2. Instale as dependências:
   ```bash
   pip install pandas numpy scikit-learn matplotlib notebook
   ```
3. Abra o Jupyter Notebook na pasta do projeto:
   ```bash
   jupyter notebook
   ```
4. Abra o arquivo `Danilo Cho - EP1-SVM-Classific_Sentimentos-2024-alunos.ipynb` e execute as células sequencialmente.
