---
title: Redacao Clara
emoji: ✍️
colorFrom: purple
colorTo: blue
sdk: gradio
sdk_version: "5.49.1"
app_file: app.py
pinned: false
license: mit
---

# Redacao Clara

Assistente RAG para entender criterios da redacao do Enem com base em documentos oficiais do Inep.

## Base de conhecimento

- Cartilha do Participante 2025.
- Manual de situacoes que levam a nota zero.
- Manuais de correcao das competencias 1 a 5.

## Tecnologias

- Python
- Gradio
- LangChain
- ChromaDB
- Sentence Transformers
- Groq
- PyPDF

## Como usar

```bash
pip install -r requirements.txt
python app.py
```

Depois, abra o link local e cole sua propria `GROQ_API_KEY`.

## Limites

Este app e um apoio de estudo. Ele nao substitui professor, corretor oficial,
simulado de redacao ou a leitura integral dos documentos do Inep.

## Arquitetura

```text
PDFs oficiais -> PyPDFLoader -> chunks -> metadados -> embeddings locais
-> ChromaDB -> retriever -> prompt -> Groq -> resposta com fontes
```
