from pathlib import Path

import gradio as gr
from google.genai import errors as genai_errors
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


APP_TITLE = "Leao Claro"
PDF_PATH = Path("P&R IRPF 2026 - v1.00 - 2026.04.23.pdf")
CHROMA_DIR = Path("chroma_leao_claro_irpf_local")
COLLECTION_NAME = "leao_claro_irpf_local"
LLM_MODEL = "gemini-2.5-flash-lite"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
RETRIEVER_K = 3
BATCH_SIZE = 80
MAX_QUESTION_CHARS = 600
SENSITIVE_TERMS = [
    "cpf",
    "senha",
    "gov.br",
    "banco",
    "agencia",
    "agência",
    "conta corrente",
    "cartao",
    "cartão",
    "pix",
]
VECTORSTORE_CACHE = {}
LLM_CACHE = {}
ANSWER_CACHE = {}

PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Voce e o Leao Claro, um assistente informativo sobre IRPF. "
            "Responda em portugues do Brasil usando somente o contexto oficial fornecido. "
            "Nao solicite CPF, senha gov.br, dados bancarios ou informacoes sensiveis. "
            "Nao prometa economia tributaria nem substitua contador ou canais oficiais da Receita Federal. "
            "Se a resposta nao estiver no contexto, diga que nao encontrou essa informacao no documento.\n\n"
            "Contexto:\n{context}",
        ),
        ("human", "{input}"),
    ]
)

CUSTOM_CSS = """
:root {
  --lc-ink: #17202a;
  --lc-muted: #667085;
  --lc-line: #e6e8ec;
  --lc-surface: #ffffff;
  --lc-soft: #f7f8fa;
  --lc-accent: #c84624;
  --lc-accent-dark: #9f3219;
  --lc-green: #1f7a4d;
}

.gradio-container {
  max-width: 1120px !important;
  margin: 0 auto !important;
  color: var(--lc-ink);
}

.lc-header {
  padding: 22px 0 14px;
  border-bottom: 1px solid var(--lc-line);
  margin-bottom: 16px;
}

.lc-kicker {
  color: var(--lc-accent);
  font-size: 13px;
  font-weight: 700;
  letter-spacing: 0;
  margin-bottom: 6px;
}

.lc-header h1 {
  font-size: 34px;
  line-height: 1.1;
  margin: 0 0 8px;
}

.lc-subtitle {
  color: var(--lc-muted);
  font-size: 16px;
  line-height: 1.5;
  max-width: 820px;
}

.lc-alert {
  border-left: 4px solid var(--lc-accent);
  background: #fff7f3;
  padding: 12px 14px;
  margin: 12px 0 18px;
  color: #4b2519;
}

.lc-section-title {
  margin: 8px 0 4px;
  font-size: 20px;
}

.lc-helper {
  color: var(--lc-muted);
  font-size: 14px;
  line-height: 1.45;
}

.lc-metric-row {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 10px;
  margin: 12px 0 18px;
}

.lc-metric {
  border: 1px solid var(--lc-line);
  background: var(--lc-soft);
  padding: 12px;
  border-radius: 6px;
}

.lc-metric strong {
  display: block;
  font-size: 15px;
  margin-bottom: 4px;
}

.lc-metric span {
  display: block;
  color: var(--lc-muted);
  font-size: 13px;
}

.lc-flow {
  border: 1px solid var(--lc-line);
  background: var(--lc-surface);
  padding: 14px;
  border-radius: 6px;
  margin-top: 10px;
}

.lc-flow code {
  white-space: normal;
}

button.primary {
  background: var(--lc-accent) !important;
  border-color: var(--lc-accent) !important;
}

button.primary:hover {
  background: var(--lc-accent-dark) !important;
  border-color: var(--lc-accent-dark) !important;
}

details {
  border: 1px solid var(--lc-line);
  border-radius: 6px;
  padding: 10px 12px;
  background: #fff;
}

summary {
  cursor: pointer;
}

@media (max-width: 760px) {
  .lc-header h1 {
    font-size: 28px;
  }

  .lc-metric-row {
    grid-template-columns: 1fr;
  }
}
"""


def _cache_key(api_key: str) -> str:
    return api_key[-8:]


def _format_status(message: str) -> str:
    return f"### Status\n\n{message}"


def _friendly_error(error: Exception) -> str:
    text = str(error)
    if "API_KEY_INVALID" in text or "API key not valid" in text:
        return (
            "A chave informada nao foi aceita pela API do Gemini. "
            "Confira se voce colou uma chave do Google AI Studio."
        )
    if "RESOURCE_EXHAUSTED" in text or "quota" in text.lower():
        return (
            "A quota gratuita do Gemini foi atingida durante a geracao da resposta. "
            "Aguarde alguns minutos e tente novamente."
        )
    if "429" in text:
        return "Muitas chamadas em pouco tempo. Aguarde alguns minutos e tente novamente."
    if isinstance(error, FileNotFoundError):
        return str(error)
    return f"Nao consegui concluir a operacao agora. Detalhe tecnico: {type(error).__name__}."


def _create_vectorstore() -> Chroma:
    if "local" in VECTORSTORE_CACHE:
        return VECTORSTORE_CACHE["local"]

    embeddings = HuggingFaceEmbeddings(
        model=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR),
    )

    VECTORSTORE_CACHE["local"] = vectorstore
    return vectorstore


def _get_llm(api_key: str) -> ChatGoogleGenerativeAI:
    key = _cache_key(api_key)
    if key not in LLM_CACHE:
        LLM_CACHE[key] = ChatGoogleGenerativeAI(
            model=LLM_MODEL,
            temperature=0,
            google_api_key=api_key,
        )
    return LLM_CACHE[key]


def _load_and_split_documents():
    if not PDF_PATH.exists():
        raise FileNotFoundError(f"Arquivo de contexto nao encontrado: {PDF_PATH}")

    documents = PyPDFLoader(str(PDF_PATH)).load()
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=200,
    ).split_documents(documents)
    return chunks


def _format_sources(docs) -> str:
    sources = []
    for i, doc in enumerate(docs, start=1):
        source = Path(doc.metadata.get("source", "documento")).name
        page = doc.metadata.get("page", "N/A")
        excerpt = doc.page_content.replace("\n", " ").strip()[:700]
        sources.append(
            f"<details><summary><strong>Trecho {i}</strong> | Fonte: "
            f"<code>{source}</code> | Pagina: <code>{page}</code></summary>\n\n"
            f"{excerpt}\n\n</details>"
        )
    return "\n\n".join(sources)


def answer_question(api_key: str, question: str):
    api_key = (api_key or "").strip()
    question = (question or "").strip()

    if not api_key:
        yield _format_status("Cole sua GOOGLE_API_KEY do Google AI Studio para testar o app.")
        return
    if not question:
        yield _format_status("Digite uma pergunta sobre IRPF.")
        return
    if len(question) > MAX_QUESTION_CHARS:
        yield _format_status(f"Sua pergunta esta muito longa. Use ate {MAX_QUESTION_CHARS} caracteres.")
        return

    lower_question = question.lower()
    if any(term in lower_question for term in SENSITIVE_TERMS):
        yield _format_status(
            "Por seguranca, nao envie CPF, senha gov.br, dados bancarios ou informacoes sensiveis. "
            "Reformule a pergunta sem dados pessoais."
        )
        return

    cache_key = (_cache_key(api_key), question.lower())
    if cache_key in ANSWER_CACHE:
        yield ANSWER_CACHE[cache_key]
        return

    yield _format_status("Preparando a base de conhecimento...")

    try:
        vectorstore = _create_vectorstore()

        saved_chunks = vectorstore._collection.count()
        if saved_chunks > 0:
            yield _format_status(
                f"Banco vetorial carregado com {saved_chunks} trechos salvos. Buscando contexto..."
            )
        else:
            yield _format_status(
                "Primeiro uso detectado: criando embeddings locais do PDF. "
                "Isso pode levar alguns minutos, mas nao consome quota de embeddings do Gemini."
            )

            chunks = _load_and_split_documents()

            total_batches = (len(chunks) + BATCH_SIZE - 1) // BATCH_SIZE
            for batch_index, start in enumerate(range(0, len(chunks), BATCH_SIZE), start=1):
                batch = chunks[start : start + BATCH_SIZE]
                vectorstore.add_documents(batch)
                indexed = min(start + BATCH_SIZE, len(chunks))
                yield _format_status(
                    f"Indexados {indexed}/{len(chunks)} trechos ({batch_index}/{total_batches})."
                )

            yield _format_status("Base indexada. Buscando contexto...")

        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": RETRIEVER_K},
        )

        docs = retriever.invoke(question)
        if not docs:
            yield _format_status("Nao encontrei trechos relevantes no documento oficial.")
            return

        context = "\n\n".join(doc.page_content for doc in docs)
        yield _format_status("Contexto encontrado. Gerando resposta com Gemini...")

        llm = _get_llm(api_key)

        messages = PROMPT.invoke({"input": question, "context": context})
        response = llm.invoke(messages)
        answer = response.content if hasattr(response, "content") else str(response)

        final_response = (
            f"## Resposta\n\n{answer}\n\n"
            "---\n\n"
            "### Fontes usadas\n\n"
            f"{_format_sources(docs)}\n\n"
            "---\n\n"
            "_Resposta gerada com base nos trechos recuperados do documento oficial. "
            "Use como orientacao inicial; em caso de duvida, consulte a Receita Federal ou um profissional._"
        )
        ANSWER_CACHE[cache_key] = final_response
        yield final_response

    except (genai_errors.APIError, Exception) as error:
        yield _format_status(_friendly_error(error))


with gr.Blocks(title=APP_TITLE, css=CUSTOM_CSS) as demo:
    gr.HTML(
        """
        <header class="lc-header">
          <div class="lc-kicker">Assistente RAG para IRPF</div>
          <h1>Leao Claro</h1>
          <p class="lc-subtitle">
            Tire duvidas gerais sobre Imposto de Renda Pessoa Fisica com respostas baseadas
            em documento oficial da Receita Federal e trechos usados como fonte.
          </p>
        </header>
        """
    )

    with gr.Tabs():
        with gr.Tab("Assistente"):
            gr.HTML(
                """
                <div class="lc-alert">
                  <strong>Uso seguro:</strong> nao informe CPF, senha gov.br, dados bancarios
                  ou informacoes pessoais. Este app e informativo e nao substitui contador,
                  consultor tributario ou canais oficiais da Receita Federal.
                </div>
                """
            )

            gr.Markdown(
                "### Faça uma pergunta\n"
                "Cole sua propria chave do Google AI Studio, escreva uma pergunta geral sobre IRPF "
                "e confira a resposta com as fontes recuperadas."
            )

            gr.Markdown(
                """
                <details>
                <summary><strong>Como conseguir uma chave gratuita do Google AI Studio?</strong></summary>

                1. Acesse https://aistudio.google.com/app/apikey
                2. Entre com sua conta Google.
                3. Clique em **Create API key** ou **Criar chave de API**.
                4. Copie a chave gerada.
                5. Cole a chave no campo abaixo.

                A chave fica oculta na tela e e usada apenas para chamar o Gemini durante o teste.
                </details>
                """
            )

            api_key = gr.Textbox(
                label="Sua GOOGLE_API_KEY",
                type="password",
                placeholder="Cole aqui sua chave do Google AI Studio",
                info="A chave e usada apenas na sua sessao para chamar o Gemini.",
            )

            question = gr.Textbox(
                label="Pergunta",
                placeholder="Ex.: O que e a declaracao pre-preenchida?",
                lines=3,
                max_lines=5,
                info="Nao inclua CPF, senha gov.br, dados bancarios ou informacoes pessoais.",
            )
            gr.Examples(
                examples=[
                    "O que e a declaracao pre-preenchida?",
                    "Quais documentos devo separar antes de declarar o IRPF?",
                    "Como funciona a restituicao do imposto de renda?",
                ],
                inputs=question,
                label="Exemplos de perguntas",
            )
            output = gr.Markdown(label="Resposta")
            with gr.Row():
                button = gr.Button("Perguntar", variant="primary")
                clear = gr.ClearButton([question, output], value="Limpar")

            button.click(
                fn=answer_question,
                inputs=[api_key, question],
                outputs=output,
            )

        with gr.Tab("Sobre o projeto"):
            gr.HTML(
                """
                <section>
                  <h2 class="lc-section-title">O que este projeto resolve</h2>
                  <p class="lc-helper">
                    Documentos de IRPF sao longos, tecnicos e dificeis de consultar.
                    O Leao Claro usa RAG para recuperar trechos oficiais antes de gerar uma resposta,
                    reduzindo o risco de respostas sem base documental.
                  </p>
                  <div class="lc-metric-row">
                    <div class="lc-metric">
                      <strong>Fonte oficial</strong>
                      <span>PDF de Perguntas e Respostas IRPF 2026 da Receita Federal.</span>
                    </div>
                    <div class="lc-metric">
                      <strong>RAG</strong>
                      <span>Busca semantica antes da resposta para trazer contexto relevante.</span>
                    </div>
                    <div class="lc-metric">
                      <strong>Embeddings locais</strong>
                      <span>Reduz consumo da quota do Gemini na criacao da base vetorial.</span>
                    </div>
                    <div class="lc-metric">
                      <strong>Fontes visiveis</strong>
                      <span>Mostra os trechos usados para sustentar a resposta.</span>
                    </div>
                  </div>
                </section>
                """
            )

            gr.HTML(
                """
                <section>
                  <h2 class="lc-section-title">Arquitetura</h2>
                  <div class="lc-flow">
                    <code>
                      PDF oficial -> PyPDFLoader -> chunks -> embeddings locais ->
                      ChromaDB -> retriever -> prompt com contexto -> Gemini -> resposta com fontes
                    </code>
                  </div>
                </section>
                """
            )

            gr.Markdown(
                """
                ### Ferramentas e métodos

                - **Python** para orquestração da aplicação.
                - **Gradio** para a interface web.
                - **LangChain** para loader, prompt e integração do fluxo RAG.
                - **ChromaDB** para persistência e busca vetorial.
                - **Sentence Transformers** para embeddings locais.
                - **Gemini** para geração da resposta final.
                - **Prompt engineering** para limitar escopo, segurança e uso das fontes.

                ### Por que isso importa

                O projeto mostra como usar IA generativa com mais responsabilidade:
                a resposta não depende apenas do conhecimento do modelo, mas de trechos recuperados
                de um documento oficial. Isso é especialmente importante em temas sensíveis como IRPF.

                ### Limitações

                - Não calcula imposto.
                - Não preenche declaração.
                - Não substitui contador ou Receita Federal.
                - Pode não encontrar a resposta se o trecho correto não for recuperado.
                - A geração final ainda depende da quota da chave Gemini informada pelo usuário.
                """
            )


if __name__ == "__main__":
    demo.queue(default_concurrency_limit=2)
    demo.launch()
