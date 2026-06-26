from pathlib import Path

import gradio as gr
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


APP_TITLE = "Bula Clara"
PDF_FILES = [Path("dipirona.pdf"), Path("paracetamol.pdf")]
CHROMA_DIR = Path("chroma_bula_clara_local")
COLLECTION_NAME = "bula_clara_local"
LLM_MODEL = "llama-3.1-8b-instant"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
RETRIEVER_K = 4
BATCH_SIZE = 80
MAX_QUESTION_CHARS = 600

VECTORSTORE_CACHE = {}
LLM_CACHE = {}
ANSWER_CACHE = {}

HIGH_RISK_TERMS = [
    "posso tomar",
    "devo tomar",
    "qual dose eu tomo",
    "qual dosagem eu tomo",
    "estou gravida",
    "gravida",
    "gestante",
    "amamentando",
    "bebe",
    "crianca",
    "alergia",
    "overdose",
    "superdosagem",
    "misturei",
    "dor no peito",
    "falta de ar",
    "desmaio",
]

CATEGORY_KEYWORDS = {
    "identificacao": ["identificacao do medicamento", "apresentacoes", "composicao"],
    "indicacao": ["para que este medicamento e indicado", "indicacoes"],
    "como_funciona": ["como este medicamento funciona"],
    "contraindicacao": ["quando nao devo usar", "contraindicacoes", "contra-indicacoes"],
    "advertencias_precaucoes": ["o que devo saber antes", "advertencias", "precaucoes"],
    "interacoes": ["interacoes medicamentosas", "interacao medicamentosa"],
    "posologia_modo_uso": ["como devo usar", "posologia", "modo de usar"],
    "reacoes_adversas": ["quais os males", "reacoes adversas", "efeitos colaterais"],
    "armazenamento": ["onde como e por quanto tempo posso guardar", "conservacao"],
    "superdosagem": ["o que fazer se alguem usar uma quantidade maior", "superdose"],
}

PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Voce e o Bula Clara, um assistente informativo de bulas de medicamentos. "
            "Responda em portugues do Brasil usando somente o contexto fornecido. "
            "Nao de diagnostico, nao recomende iniciar, parar ou alterar medicamento, "
            "e nao substitua medico, farmaceutico ou a bula oficial. "
            "Quando houver risco, duvida clinica, gestacao, criancas, alergia, mistura de remedios "
            "ou sintomas graves, oriente procurar um profissional de saude. "
            "Se a resposta nao estiver no contexto, diga que nao encontrou essa informacao nas bulas carregadas.\n\n"
            "Contexto:\n{context}",
        ),
        ("human", "{input}"),
    ]
)

CUSTOM_CSS = """
:root {
  --bc-ink: #14211f;
  --bc-muted: #66736f;
  --bc-line: #dfe7e3;
  --bc-surface: #ffffff;
  --bc-soft: #f3f7f5;
  --bc-accent: #1f6f55;
  --bc-accent-dark: #164f3d;
  --bc-mint: #dff2eb;
  --bc-blue: #2c5f8f;
  --bc-warn: #9a5b00;
  --bc-warn-soft: #fff5df;
  --bc-shadow: 0 18px 42px rgba(20, 33, 31, 0.08);
}

body,
.gradio-container {
  background: #f6f8f7 !important;
}

.gradio-container {
  max-width: 1180px !important;
  margin: 0 auto !important;
  color: var(--bc-ink);
  padding: 24px 18px 42px !important;
}

.gradio-container .contain {
  gap: 16px !important;
}

.bc-header {
  padding: 26px;
  margin-bottom: 16px;
  border: 1px solid #cfe1da;
  border-radius: 8px;
  background:
    linear-gradient(135deg, rgba(31, 111, 85, 0.96), rgba(30, 78, 89, 0.96)),
    #1f6f55;
  box-shadow: var(--bc-shadow);
  color: #ffffff;
}

.bc-kicker {
  display: inline-flex;
  width: fit-content;
  padding: 5px 9px;
  border: 1px solid rgba(255, 255, 255, 0.26);
  border-radius: 999px;
  background: rgba(255, 255, 255, 0.12);
  color: #e8fff7;
  font-size: 13px;
  font-weight: 700;
  margin-bottom: 12px;
}

.bc-header h1 {
  font-size: 36px;
  line-height: 1.1;
  margin: 0 0 10px;
  color: #ffffff;
}

.bc-subtitle {
  color: #e9f7f2;
  font-size: 16px;
  line-height: 1.5;
  max-width: 780px;
  margin: 0;
}

.bc-header-grid {
  display: grid;
  grid-template-columns: minmax(0, 1fr) auto;
  gap: 20px;
  align-items: end;
}

.bc-header-badges {
  display: grid;
  gap: 8px;
  min-width: 220px;
}

.bc-badge {
  display: flex;
  justify-content: space-between;
  gap: 18px;
  padding: 8px 10px;
  border: 1px solid rgba(255, 255, 255, 0.22);
  border-radius: 6px;
  background: rgba(255, 255, 255, 0.1);
  color: #ffffff;
  font-size: 13px;
}

.bc-badge span {
  color: #cce8df;
}

.bc-badge strong {
  font-weight: 700;
}

.bc-panel {
  border: 1px solid var(--bc-line) !important;
  border-radius: 8px !important;
  background: var(--bc-surface) !important;
  box-shadow: var(--bc-shadow);
  padding: 16px !important;
}

.bc-alert {
  border-left: 4px solid var(--bc-warn);
  background: var(--bc-warn-soft);
  padding: 12px 14px;
  margin: 0 0 12px;
  border-radius: 6px;
  color: #4a2d00;
  font-size: 14px;
  line-height: 1.45;
}

.bc-card-title {
  margin: 0 0 6px;
  font-size: 18px;
  line-height: 1.25;
  color: var(--bc-ink);
}

.bc-card-copy {
  margin: 0 0 12px;
  color: var(--bc-muted);
  font-size: 14px;
  line-height: 1.45;
}

.bc-section-title {
  margin: 8px 0 4px;
  font-size: 20px;
}

.bc-helper {
  color: var(--bc-muted);
  font-size: 14px;
  line-height: 1.45;
}

.bc-metric-row {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 12px;
  margin: 16px 0 18px;
}

.bc-metric {
  border: 1px solid var(--bc-line);
  background: var(--bc-soft);
  padding: 14px;
  border-radius: 8px;
}

.bc-metric strong {
  display: block;
  font-size: 15px;
  margin-bottom: 4px;
}

.bc-metric span {
  display: block;
  color: var(--bc-muted);
  font-size: 13px;
}

.bc-flow {
  border: 1px solid var(--bc-line);
  background: #f9fbfa;
  padding: 16px;
  border-radius: 8px;
  margin-top: 10px;
}

.bc-flow code {
  white-space: normal;
  color: #244842;
}

.bc-case-section {
  margin-bottom: 16px;
}

.bc-section-badge {
  display: inline-flex;
  width: fit-content;
  padding: 5px 10px;
  border: 1px solid #b9d8ce;
  border-radius: 999px;
  background: var(--bc-mint);
  color: var(--bc-accent-dark);
  font-size: 12px;
  font-weight: 800;
  margin-bottom: 10px;
}

.bc-case-section h2 {
  margin: 0 0 8px;
  color: var(--bc-ink);
  font-size: 24px;
  line-height: 1.18;
}

.bc-case-section h3 {
  margin: 18px 0 8px;
  color: var(--bc-ink);
  font-size: 17px;
}

.bc-case-section p {
  margin: 0 0 12px;
  color: var(--bc-muted);
  line-height: 1.55;
}

.bc-stat-grid {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 12px;
  margin-top: 16px;
}

.bc-stat {
  border: 1px solid var(--bc-line);
  border-radius: 8px;
  background: #f9fbfa;
  padding: 14px;
}

.bc-stat-num {
  display: block;
  color: var(--bc-accent-dark);
  font-size: 25px;
  font-weight: 800;
  line-height: 1;
  margin-bottom: 6px;
}

.bc-stat-label {
  display: block;
  color: var(--bc-muted);
  font-size: 13px;
  line-height: 1.35;
}

.bc-pipeline {
  display: grid;
  gap: 12px;
  margin-top: 14px;
}

.bc-pipeline-step {
  display: grid;
  grid-template-columns: 110px minmax(0, 1fr);
  gap: 14px;
  align-items: start;
  border: 1px solid var(--bc-line);
  border-radius: 8px;
  background: #fbfdfc;
  padding: 14px;
}

.bc-step-label {
  display: inline-flex;
  justify-content: center;
  align-items: center;
  min-height: 34px;
  border-radius: 6px;
  background: #e6f4ef;
  color: var(--bc-accent-dark);
  font-size: 12px;
  font-weight: 800;
  text-transform: uppercase;
}

.bc-pipeline-step h4 {
  margin: 0 0 5px;
  color: var(--bc-ink);
  font-size: 15px;
}

.bc-pipeline-step p {
  margin: 0;
  color: var(--bc-muted);
  font-size: 14px;
  line-height: 1.45;
}

.bc-method-grid {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 12px;
  margin-top: 14px;
}

.bc-method-card {
  border: 1px solid var(--bc-line);
  border-radius: 8px;
  background: #f9fbfa;
  padding: 14px;
}

.bc-method-card strong {
  display: block;
  margin-bottom: 6px;
  color: var(--bc-ink);
}

.bc-method-card span {
  color: var(--bc-muted);
  font-size: 13px;
  line-height: 1.4;
}

.bc-bar-list {
  display: grid;
  gap: 12px;
  margin-top: 14px;
}

.bc-bar-row {
  display: grid;
  gap: 6px;
}

.bc-bar-head {
  display: flex;
  justify-content: space-between;
  gap: 12px;
  color: var(--bc-ink);
  font-size: 14px;
  font-weight: 700;
}

.bc-bar-track {
  height: 9px;
  overflow: hidden;
  border-radius: 999px;
  background: #e7eeeb;
}

.bc-bar-fill {
  display: block;
  height: 100%;
  border-radius: inherit;
  background: linear-gradient(90deg, var(--bc-accent), var(--bc-blue));
}

.bc-limit-list {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 10px;
  margin-top: 14px;
}

.bc-limit {
  border-left: 4px solid var(--bc-warn);
  border-radius: 6px;
  background: var(--bc-warn-soft);
  padding: 12px;
  color: #4a2d00;
  font-size: 14px;
  line-height: 1.4;
}

.bc-side-note {
  border: 1px solid #cfe1da;
  border-radius: 8px;
  background: #f8fcfa;
  padding: 14px;
  color: var(--bc-muted);
  font-size: 14px;
  line-height: 1.45;
}

.bc-side-note strong {
  display: block;
  margin-bottom: 6px;
  color: var(--bc-ink);
}

.bc-answer {
  margin-top: 14px;
}

.bc-answer > div {
  border-radius: 8px !important;
}

.bc-actions button {
  min-height: 44px !important;
  border-radius: 6px !important;
  font-weight: 700 !important;
}

button.primary {
  background: var(--bc-accent) !important;
  border-color: var(--bc-accent) !important;
}

button.primary:hover {
  background: var(--bc-accent-dark) !important;
  border-color: var(--bc-accent-dark) !important;
}

details {
  border: 1px solid var(--bc-line);
  border-radius: 8px;
  padding: 10px 12px;
  background: #fff;
}

summary {
  cursor: pointer;
}

textarea,
input {
  border-radius: 6px !important;
}

.tabs {
  border: 1px solid var(--bc-line);
  border-radius: 8px !important;
  background: var(--bc-surface);
  box-shadow: var(--bc-shadow);
  padding: 8px !important;
}

.tab-nav {
  border-bottom: 1px solid var(--bc-line) !important;
}

@media (max-width: 760px) {
  .bc-header h1 {
    font-size: 28px;
  }

  .bc-header {
    padding: 20px;
  }

  .bc-header-grid {
    grid-template-columns: 1fr;
  }

  .bc-header-badges {
    min-width: 0;
  }

  .bc-metric-row {
    grid-template-columns: 1fr;
  }

  .bc-stat-grid,
  .bc-method-grid,
  .bc-limit-list {
    grid-template-columns: 1fr;
  }

  .bc-pipeline-step {
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
    if "invalid_api_key" in text.lower() or "api key" in text.lower():
        return "A chave informada nao foi aceita pela API da Groq. Confira se voce colou a GROQ_API_KEY correta."
    if "rate_limit" in text.lower() or "429" in text:
        return "A API da Groq limitou muitas chamadas em pouco tempo. Aguarde um pouco e tente novamente."
    if isinstance(error, FileNotFoundError):
        return str(error)
    return f"Nao consegui concluir agora. Detalhe tecnico: {type(error).__name__}."


def _infer_medicine_name(path: Path) -> str:
    return path.stem.replace("_", " ").replace("-", " ").title()


def _classify_category(text: str) -> str:
    normalized = text.lower()
    for category, keywords in CATEGORY_KEYWORDS.items():
        if any(keyword in normalized for keyword in keywords):
            return category
    return "geral"


def _load_and_split_documents():
    missing = [str(path) for path in PDF_FILES if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Nao encontrei estes PDFs: {', '.join(missing)}")

    documents = []
    for pdf_path in PDF_FILES:
        loaded_docs = PyPDFLoader(str(pdf_path)).load()
        medicine = _infer_medicine_name(pdf_path)
        for doc in loaded_docs:
            doc.metadata["medicamento"] = medicine
            doc.metadata["source"] = pdf_path.name
        documents.extend(loaded_docs)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=160,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)

    for chunk in chunks:
        chunk.metadata["categoria"] = _classify_category(chunk.page_content)

    return chunks


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


def _get_llm(api_key: str) -> ChatGroq:
    key = _cache_key(api_key)
    if key not in LLM_CACHE:
        LLM_CACHE[key] = ChatGroq(
            groq_api_key=api_key,
            model=LLM_MODEL,
            temperature=0,
        )
    return LLM_CACHE[key]


def _format_sources(docs) -> str:
    sources = []
    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "PDF")
        page = doc.metadata.get("page", "?")
        medicine = doc.metadata.get("medicamento", "Medicamento")
        category = doc.metadata.get("categoria", "geral")
        excerpt = doc.page_content.replace("\n", " ").strip()[:700]
        sources.append(
            f"<details><summary><strong>Trecho {i}</strong> | "
            f"<code>{medicine}</code> | <code>{category}</code> | "
            f"Fonte: <code>{source}</code> | Pagina: <code>{page}</code></summary>\n\n"
            f"{excerpt}\n\n</details>"
        )
    return "\n\n".join(sources)


def _risk_notice(question: str) -> str:
    lower_question = question.lower()
    if any(term in lower_question for term in HIGH_RISK_TERMS):
        return (
            "\n\n> **Aviso de seguranca:** sua pergunta parece envolver decisao clinica, dose, "
            "risco, sintoma ou situacao individual. Use a resposta apenas como leitura inicial da bula "
            "e procure um medico ou farmaceutico antes de agir."
        )
    return ""


def answer_question(api_key: str, question: str):
    api_key = (api_key or "").strip()
    question = (question or "").strip()

    if not api_key:
        yield _format_status("Cole sua GROQ_API_KEY para testar o app.")
        return
    if not question:
        yield _format_status("Digite uma pergunta geral sobre as bulas carregadas.")
        return
    if len(question) > MAX_QUESTION_CHARS:
        yield _format_status(f"Sua pergunta esta muito longa. Use ate {MAX_QUESTION_CHARS} caracteres.")
        return

    cache_key = (_cache_key(api_key), question.lower())
    if cache_key in ANSWER_CACHE:
        yield ANSWER_CACHE[cache_key]
        return

    yield _format_status("Preparando a base de conhecimento das bulas...")

    try:
        vectorstore = _create_vectorstore()

        saved_chunks = vectorstore._collection.count()
        if saved_chunks > 0:
            yield _format_status(f"Banco vetorial carregado com {saved_chunks} trechos salvos. Buscando contexto...")
        else:
            yield _format_status(
                "Primeiro uso detectado: criando embeddings locais das bulas. "
                "Isso pode levar alguns minutos, mas nao consome tokens da Groq."
            )
            chunks = _load_and_split_documents()

            total_batches = (len(chunks) + BATCH_SIZE - 1) // BATCH_SIZE
            for batch_index, start in enumerate(range(0, len(chunks), BATCH_SIZE), start=1):
                batch = chunks[start : start + BATCH_SIZE]
                vectorstore.add_documents(batch)
                indexed = min(start + BATCH_SIZE, len(chunks))
                yield _format_status(f"Indexados {indexed}/{len(chunks)} trechos ({batch_index}/{total_batches}).")

            yield _format_status("Base indexada. Buscando contexto relevante...")

        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": RETRIEVER_K},
        )
        docs = retriever.invoke(question)

        if not docs:
            yield _format_status("Nao encontrei trechos relevantes nas bulas carregadas.")
            return

        context = "\n\n".join(
            f"[{doc.metadata.get('medicamento', 'Medicamento')} | "
            f"{doc.metadata.get('categoria', 'geral')} | "
            f"pagina {doc.metadata.get('page', '?')}]\n{doc.page_content}"
            for doc in docs
        )

        yield _format_status("Contexto encontrado. Gerando resposta com Groq...")

        llm = _get_llm(api_key)
        messages = PROMPT.invoke({"input": question, "context": context})
        response = llm.invoke(messages)
        answer = response.content if hasattr(response, "content") else str(response)

        final_response = (
            f"## Resposta\n\n{answer}"
            f"{_risk_notice(question)}\n\n"
            "---\n\n"
            "### Fontes usadas\n\n"
            f"{_format_sources(docs)}\n\n"
            "---\n\n"
            "_Resposta gerada com base nos trechos recuperados das bulas carregadas. "
            "Nao use este app para diagnostico, automedicacao ou decisao de dose._"
        )
        ANSWER_CACHE[cache_key] = final_response
        yield final_response

    except Exception as error:
        yield _format_status(_friendly_error(error))


with gr.Blocks(title=APP_TITLE, css=CUSTOM_CSS) as demo:
    gr.HTML(
        """
        <header class="bc-header">
          <div class="bc-header-grid">
            <div>
              <div class="bc-kicker">RAG aplicado a bulas</div>
              <h1>Bula Clara</h1>
              <p class="bc-subtitle">
                Perguntas gerais sobre dipirona e paracetamol, com resposta baseada
                nos PDFs carregados e fontes abertas para conferencia.
              </p>
            </div>
            <div class="bc-header-badges" aria-label="Resumo tecnico do projeto">
              <div class="bc-badge"><span>LLM</span><strong>Groq</strong></div>
              <div class="bc-badge"><span>Busca</span><strong>ChromaDB</strong></div>
              <div class="bc-badge"><span>Embeddings</span><strong>Locais</strong></div>
            </div>
          </div>
        </header>
        """
    )

    with gr.Tabs():
        with gr.Tab("Assistente"):
            with gr.Row():
                with gr.Column(scale=7, elem_classes=["bc-panel"]):
                    gr.HTML(
                        """
                        <h2 class="bc-card-title">Consulta</h2>
                        <p class="bc-card-copy">
                          Use perguntas gerais sobre as bulas carregadas. A resposta mostra os trechos usados.
                        </p>
                        """
                    )

                    api_key = gr.Textbox(
                        label="Sua GROQ_API_KEY",
                        type="password",
                        placeholder="Cole aqui sua chave da Groq",
                        info="Usada somente nesta sessao para chamar a Groq.",
                    )

                    question = gr.Textbox(
                        label="Pergunta",
                        placeholder="Ex.: Quais sao as principais contraindicacoes da dipirona?",
                        lines=4,
                        max_lines=6,
                        info="Para decisao medica individual, procure um profissional.",
                    )

                    with gr.Row(elem_classes=["bc-actions"]):
                        button = gr.Button("Perguntar", variant="primary")
                        clear = gr.ClearButton([question], value="Limpar pergunta")

                with gr.Column(scale=4, elem_classes=["bc-panel"]):
                    gr.HTML(
                        """
                        <div class="bc-alert">
                          <strong>Uso seguro</strong><br>
                          Nao use para decidir dose, misturar remedios, interromper tratamento
                          ou lidar com sintomas graves.
                        </div>
                        <div class="bc-side-note">
                          <strong>Chave da Groq</strong>
                          Acesse <code>console.groq.com/keys</code>, crie uma API key e cole no campo ao lado.
                          O plano gratuito pode ter limites de uso e velocidade.
                        </div>
                        """
                    )

            gr.Examples(
                examples=[
                    "Para que a dipirona e indicada?",
                    "Quais sao as principais contraindicacoes do paracetamol?",
                    "O que a bula diz sobre reacoes adversas da dipirona?",
                    "Como o paracetamol deve ser armazenado?",
                    "O que fazer em caso de superdosagem segundo a bula?",
                ],
                inputs=question,
                label="Perguntas de teste",
            )

            output = gr.Markdown(label="Resposta", elem_classes=["bc-answer"])

            button.click(
                fn=answer_question,
                inputs=[api_key, question],
                outputs=output,
            )

        with gr.Tab("Sobre o projeto"):
            gr.HTML(
                """
                <section class="bc-panel bc-case-section">
                  <div class="bc-section-badge">Fase 1</div>
                  <h2>Problema e proposta</h2>
                  <p>
                    Bulas sao documentos importantes, mas longos, tecnicos e pouco amigaveis para
                    uma consulta rapida. O Bula Clara transforma essa leitura em uma conversa
                    com respostas baseadas nos PDFs carregados, mantendo os trechos usados visiveis.
                  </p>
                  <div class="bc-stat-grid">
                    <div class="bc-stat">
                      <span class="bc-stat-num">2</span>
                      <span class="bc-stat-label">bulas usadas como base inicial</span>
                    </div>
                    <div class="bc-stat">
                      <span class="bc-stat-num">4</span>
                      <span class="bc-stat-label">trechos recuperados por pergunta</span>
                    </div>
                    <div class="bc-stat">
                      <span class="bc-stat-num">0</span>
                      <span class="bc-stat-label">tokens gastos para embeddings em API</span>
                    </div>
                    <div class="bc-stat">
                      <span class="bc-stat-num">100%</span>
                      <span class="bc-stat-label">das respostas exibem fontes recuperadas</span>
                    </div>
                  </div>
                </section>
                """
            )

            gr.HTML(
                """
                <section class="bc-panel bc-case-section">
                  <div class="bc-section-badge">Fase 2</div>
                  <h2>Pipeline RAG</h2>
                  <p>
                    O fluxo evita mandar a pergunta diretamente para a LLM. Primeiro o sistema busca
                    contexto nos documentos; depois a resposta e gerada usando apenas esse material.
                  </p>
                  <div class="bc-pipeline">
                    <div class="bc-pipeline-step">
                      <span class="bc-step-label">Entrada</span>
                      <div>
                        <h4>PDFs das bulas</h4>
                        <p>Dipirona e paracetamol sao carregados com PyPDFLoader e recebem metadados de origem.</p>
                      </div>
                    </div>
                    <div class="bc-pipeline-step">
                      <span class="bc-step-label">Texto</span>
                      <div>
                        <h4>Chunks com sobreposicao</h4>
                        <p>O conteudo e dividido em trechos menores para melhorar a busca e preservar contexto.</p>
                      </div>
                    </div>
                    <div class="bc-pipeline-step">
                      <span class="bc-step-label">Vetores</span>
                      <div>
                        <h4>Embeddings locais</h4>
                        <p>Sentence Transformers cria representacoes semanticas sem consumir quota da Groq.</p>
                      </div>
                    </div>
                    <div class="bc-pipeline-step">
                      <span class="bc-step-label">Busca</span>
                      <div>
                        <h4>ChromaDB e retriever</h4>
                        <p>Os vetores ficam persistidos e a busca retorna os trechos mais relevantes para a pergunta.</p>
                      </div>
                    </div>
                    <div class="bc-pipeline-step">
                      <span class="bc-step-label">Resposta</span>
                      <div>
                        <h4>Groq com prompt seguro</h4>
                        <p>A LLM gera uma explicacao em portugues e o app exibe os trechos usados como fonte.</p>
                      </div>
                    </div>
                  </div>
                </section>
                """
            )

            gr.HTML(
                """
                <section class="bc-panel bc-case-section">
                  <div class="bc-section-badge">Fase 3</div>
                  <h2>Ferramentas e decisoes tecnicas</h2>
                  <p>
                    A arquitetura separa o que precisa rodar localmente do que depende de inferencia em cloud.
                    Isso reduz custo, melhora previsibilidade e deixa o projeto mais facil de demonstrar.
                  </p>
                  <div class="bc-method-grid">
                    <div class="bc-method-card">
                      <strong>Interface</strong>
                      <span>Gradio organiza a experiencia em abas, exemplos, campo de chave e resposta com fontes.</span>
                    </div>
                    <div class="bc-method-card">
                      <strong>Orquestracao</strong>
                      <span>LangChain conecta loader, splitter, embeddings, prompt e chamada da LLM.</span>
                    </div>
                    <div class="bc-method-card">
                      <strong>Busca vetorial</strong>
                      <span>ChromaDB persiste os embeddings para evitar reindexar a cada execucao.</span>
                    </div>
                    <div class="bc-method-card">
                      <strong>Embeddings</strong>
                      <span>Sentence Transformers roda localmente e funciona bem para texto em portugues.</span>
                    </div>
                    <div class="bc-method-card">
                      <strong>LLM</strong>
                      <span>Groq gera a resposta final em cloud usando a chave informada pelo usuario.</span>
                    </div>
                    <div class="bc-method-card">
                      <strong>Seguranca</strong>
                      <span>O prompt limita diagnostico, dose individual e decisoes medicas sensiveis.</span>
                    </div>
                  </div>
                </section>
                """
            )

            gr.HTML(
                """
                <section class="bc-panel bc-case-section">
                  <div class="bc-section-badge">Fase 4</div>
                  <h2>Qualidade do produto</h2>
                  <p>
                    Para portfolio, o ponto forte nao e apenas responder. E mostrar que a aplicacao foi pensada
                    com transparencia, custo controlado e limites claros para um tema sensivel.
                  </p>
                  <div class="bc-bar-list">
                    <div class="bc-bar-row">
                      <div class="bc-bar-head"><span>Fontes auditaveis</span><span>Alta</span></div>
                      <div class="bc-bar-track"><span class="bc-bar-fill" style="width: 96%;"></span></div>
                    </div>
                    <div class="bc-bar-row">
                      <div class="bc-bar-head"><span>Custo de indexacao</span><span>Baixo</span></div>
                      <div class="bc-bar-track"><span class="bc-bar-fill" style="width: 88%;"></span></div>
                    </div>
                    <div class="bc-bar-row">
                      <div class="bc-bar-head"><span>Clareza para o usuario</span><span>Alta</span></div>
                      <div class="bc-bar-track"><span class="bc-bar-fill" style="width: 90%;"></span></div>
                    </div>
                    <div class="bc-bar-row">
                      <div class="bc-bar-head"><span>Escalabilidade para novos PDFs</span><span>Media-alta</span></div>
                      <div class="bc-bar-track"><span class="bc-bar-fill" style="width: 78%;"></span></div>
                    </div>
                  </div>
                </section>
                """
            )

            gr.HTML(
                """
                <section class="bc-panel bc-case-section">
                  <div class="bc-section-badge">Limites</div>
                  <h2>Uso responsavel</h2>
                  <p>
                    Como o tema envolve saude, a interface precisa comunicar claramente o que o sistema nao faz.
                  </p>
                  <div class="bc-limit-list">
                    <div class="bc-limit">Nao fornece diagnostico ou avaliacao clinica individual.</div>
                    <div class="bc-limit">Nao recomenda iniciar, parar ou alterar medicamento.</div>
                    <div class="bc-limit">Nao substitui medico, farmaceutico ou bula oficial.</div>
                    <div class="bc-limit">Pode falhar se o trecho correto nao for recuperado pelo retriever.</div>
                  </div>
                </section>
                """
            )


if __name__ == "__main__":
    demo.queue(default_concurrency_limit=2)
    demo.launch()
