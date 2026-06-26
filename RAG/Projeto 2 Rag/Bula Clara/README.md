---
title: Bula Clara
emoji: 💊
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: "5.49.1"
app_file: app.py
pinned: false
license: mit
---

# Bula Clara

Assistente RAG para consulta explicativa de bulas de dipirona e paracetamol.
O app recupera trechos dos PDFs carregados antes de gerar a resposta, mostrando as fontes usadas.

## Problema

Bulas sao documentos importantes, mas podem ser longas, tecnicas e dificeis de consultar.
O Bula Clara transforma essa leitura em uma conversa guiada, sem esconder os trechos usados como base.

## Solucao

O projeto usa RAG:

```text
PDFs das bulas -> PyPDFLoader -> chunks -> metadados -> embeddings locais
-> ChromaDB -> retriever -> prompt seguro -> Groq -> resposta com fontes
```

## Tecnologias

- Python
- Gradio
- LangChain
- ChromaDB
- Sentence Transformers
- Groq
- PyPDF

## Como usar

1. Instale as dependencias:

```bash
pip install -r requirements.txt
```

2. Rode o app:

```bash
python app.py
```

3. Abra o link local exibido no terminal.
4. Cole sua propria `GROQ_API_KEY`.
5. Pergunte algo geral sobre as bulas carregadas.

## Como obter uma chave da Groq

1. Acesse https://console.groq.com/keys
2. Entre ou crie uma conta.
3. Crie uma API key.
4. Copie a chave.
5. Cole no campo do app.

O plano gratuito pode ter limites de uso e velocidade. A chave e usada somente para gerar a resposta final.
Os embeddings sao locais e nao consomem tokens da Groq.

## Seguranca medica

Este projeto e informativo. Ele nao substitui medico, farmaceutico ou a bula oficial.
Nao use o app para diagnostico, automedicacao, decisao de dose, mistura de medicamentos
ou interrupcao de tratamento.

## Por que isso e bom para portfolio

O projeto demonstra:

- Uso de RAG com documentos reais.
- Persistencia de embeddings com ChromaDB.
- Separacao entre indexacao local e inferencia cloud.
- Interface com fontes visiveis.
- Prompt engineering para uma area sensivel.
- Produto com limites claros e foco em utilidade real.

## Limites conhecidos

- A qualidade depende dos PDFs carregados.
- A busca pode recuperar trechos incompletos em perguntas muito amplas.
- A resposta final depende dos limites da Groq.
- O app nao interpreta exames, sintomas ou casos individuais.
