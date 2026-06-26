---
title: Leao Claro
emoji: 🦁
colorFrom: yellow
colorTo: red
sdk: gradio
sdk_version: "5.49.1"
app_file: app.py
pinned: false
license: mit
---

# Leao Claro

Assistente RAG para tirar duvidas sobre IRPF com base em documento oficial da Receita Federal.

> Projeto educacional e informativo. Nao substitui contador, advogado, consultor tributario nem canais oficiais da Receita Federal.

## Problema

Declarar o Imposto de Renda pode ser confuso: o contribuinte precisa navegar por documentos longos, termos tecnicos e regras especificas.

LLMs podem ajudar, mas tambem podem alucinar quando respondem sem fonte. O Leao Claro usa RAG para reduzir esse risco: antes de responder, o app recupera trechos relevantes do documento oficial e usa esses trechos como contexto.

## Solucao

O usuario faz uma pergunta geral sobre IRPF. O app:

1. Carrega uma base oficial da Receita Federal.
2. Recupera os trechos mais relevantes com ChromaDB.
3. Envia pergunta e contexto ao Gemini.
4. Mostra a resposta e as fontes usadas.

O app nao pede CPF, senha gov.br, dados bancarios ou informacoes sensiveis.

## Interface

A aplicacao tem duas areas principais:

- **Assistente:** tela de uso do produto, com campo de chave, pergunta, exemplos, resposta e fontes.
- **Sobre o projeto:** pagina de apresentacao para portfolio, explicando problema, arquitetura, ferramentas, importancia e limitacoes.

A UI usa mensagens progressivas de status para deixar claro quando a base esta sendo carregada, indexada ou consultada.

## Tecnologias

- Python
- Gradio
- LangChain
- Gemini API
- ChromaDB
- Embeddings locais com Sentence Transformers
- PDF document loading
- Retrieval-Augmented Generation
- Prompt engineering

## Arquitetura

```text
PDF oficial da Receita
        |
        v
PyPDFLoader -> text splitter -> embeddings locais
        |
        v
ChromaDB persistido em disco
        |
        v
Retriever -> prompt com contexto -> Gemini -> resposta com fontes
```

## Como Usar

Na interface:

1. Cole sua propria `GOOGLE_API_KEY` do Google AI Studio.
2. Digite uma pergunta geral sobre IRPF.
3. Confira a resposta e os trechos usados como fonte.

Crie uma chave no Google AI Studio:

https://aistudio.google.com/app/apikey

## Como Obter Uma Chave do Google AI Studio

1. Acesse https://aistudio.google.com/app/apikey.
2. Entre com sua conta Google.
3. Clique em **Create API key** ou **Criar chave de API**.
4. Copie a chave gerada.
5. Cole a chave no campo `Sua GOOGLE_API_KEY` dentro do app.

A chave fica oculta na interface e e usada apenas para chamar o Gemini durante o teste. Nao publique sua chave em GitHub, Hugging Face, prints ou mensagens.

## Por Que o Usuario Usa a Propria Chave?

Para evitar expor ou consumir a chave do autor do projeto.

Cada usuario usa a propria quota gratuita/paga do Google AI Studio para gerar as respostas. A chave e digitada na interface e nao fica salva no codigo do projeto.

## Como Rodar Localmente

Instale as dependencias:

```bash
pip install -r requirements.txt
```

Execute o app:

```bash
python app.py
```

## Como Publicar no Hugging Face Spaces

Crie um Space com SDK **Gradio** e suba estes arquivos:

- `app.py`
- `requirements.txt`
- `README.md`
- `P&R IRPF 2026 - v1.00 - 2026.04.23.pdf`

Nao suba `.env`.

## Caching e Custo

O ChromaDB salva os embeddings locais na pasta:

```text
chroma_leao_claro_irpf_local
```

Se essa pasta ja existir, o app carrega o banco vetorial salvo. Se nao existir, ele cria os embeddings localmente usando Sentence Transformers. Essa etapa pode demorar no primeiro uso, mas nao consome quota de embeddings do Gemini.

O app tambem usa cache em memoria para:

- reaproveitar o vectorstore durante a sessao;
- reaproveitar o modelo LLM;
- devolver rapidamente perguntas repetidas.

Mesmo com embeddings salvos, cada nova pergunta ainda usa a chave informada pelo usuario para chamar o Gemini e gerar a resposta final.

## Estados Tratados

O app exibe mensagens amigaveis para:

- chave ausente;
- pergunta vazia;
- pergunta com dados sensiveis;
- documento ausente;
- quota gratuita excedida;
- chave invalida;
- primeira indexacao da base;
- carregamento de banco vetorial salvo.

## Limitacoes

- Nao calcula imposto.
- Nao preenche declaracao.
- Nao substitui contador.
- Nao substitui canais oficiais da Receita Federal.
- Responde somente com base no documento usado como fonte.
- Pode falhar se a quota da API do usuario acabar.

## Notebook

O notebook `Leao Claro - Assistente IRPF.ipynb` mostra o processo completo:

- carregamento do PDF;
- divisao em chunks;
- criacao de embeddings;
- persistencia com ChromaDB;
- recuperacao de contexto;
- respostas com trechos usados como fonte.

## Licenca

Este projeto e distribuido sob licenca MIT.

Os documentos oficiais utilizados como fonte pertencem aos seus respectivos orgaos/autores e sao usados apenas como base publica de consulta.
