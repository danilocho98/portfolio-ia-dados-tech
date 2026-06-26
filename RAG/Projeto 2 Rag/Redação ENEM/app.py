from pathlib import Path

import gradio as gr
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


APP_TITLE = "Redacao Clara"
PDF_FILES = [
    Path("a_redacao_no_enem_2025_cartilha_do_participante.pdf"),
    Path("Situacoes_nota_zero.pdf"),
    Path("Competencia_1.pdf"),
    Path("Competencia_2.pdf"),
    Path("Competencia_3.pdf"),
    Path("Competencia_4.pdf"),
    Path("Competencia_5.pdf"),
]
CHROMA_DIR = Path("chroma_redacao_clara_local")
COLLECTION_NAME = "redacao_clara_local"
LLM_MODEL = "llama-3.1-8b-instant"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
RETRIEVER_K = 5
BATCH_SIZE = 80
MAX_QUESTION_CHARS = 800

VECTORSTORE_CACHE = {}
LLM_CACHE = {}
ANSWER_CACHE = {}

CATEGORY_KEYWORDS = {
    "competencia_1": ["competencia i", "competencia 1", "modalidade escrita formal", "desvios gramaticais"],
    "competencia_2": ["competencia ii", "competencia 2", "compreender a proposta", "dissertativo-argumentativo"],
    "competencia_3": ["competencia iii", "competencia 3", "selecionar", "relacionar", "organizar"],
    "competencia_4": ["competencia iv", "competencia 4", "mecanismos linguisticos", "coesao"],
    "competencia_5": ["competencia v", "competencia 5", "proposta de intervencao", "direitos humanos"],
    "nota_zero": ["nota zero", "fuga ao tema", "texto insuficiente", "desrespeito a seriedade"],
    "avaliacao": ["como a redacao sera avaliada", "avaliadores", "nota final", "1000 pontos"],
    "exemplos": ["redacoes nota", "amostra", "comentada", "exemplo"],
}

PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Voce e o Redacao Clara, um assistente de estudo sobre a redacao do Enem. "
            "Responda em portugues do Brasil usando somente o contexto oficial fornecido. "
            "Explique criterios, competencias e situacoes de nota zero de forma simples e pratica. "
            "Nao prometa nota, nao corrija como avaliador oficial e nao substitua professor, corretor "
            "ou a leitura da cartilha do Inep. "
            "Se a resposta nao estiver no contexto, diga que nao encontrou essa informacao nos documentos carregados.\n\n"
            "Contexto:\n{context}",
        ),
        ("human", "{input}"),
    ]
)

CUSTOM_CSS = """
:root {
  --rc-ink: #1f2430;
  --rc-muted: #697386;
  --rc-line: #e4e8ef;
  --rc-surface: #ffffff;
  --rc-soft: #f6f8fb;
  --rc-accent: #6f3fd6;
  --rc-accent-dark: #5328a5;
  --rc-blue: #2563a8;
  --rc-warn: #9a5b00;
  --rc-warn-soft: #fff5df;
  --rc-shadow: 0 18px 42px rgba(31, 36, 48, 0.08);
}

body,
.gradio-container {
  background: #f7f8fb !important;
}

.gradio-container {
  max-width: 1180px !important;
  margin: 0 auto !important;
  color: var(--rc-ink);
  padding: 24px 18px 42px !important;
}

.rc-header {
  padding: 26px;
  margin-bottom: 16px;
  border: 1px solid #d8dff0;
  border-radius: 8px;
  background:
    linear-gradient(135deg, rgba(111, 63, 214, 0.96), rgba(37, 99, 168, 0.96)),
    #6f3fd6;
  box-shadow: var(--rc-shadow);
  color: #ffffff;
}

.rc-header-grid {
  display: grid;
  grid-template-columns: minmax(0, 1fr) auto;
  gap: 20px;
  align-items: end;
}

.rc-kicker {
  display: inline-flex;
  width: fit-content;
  padding: 5px 9px;
  border: 1px solid rgba(255, 255, 255, 0.26);
  border-radius: 999px;
  background: rgba(255, 255, 255, 0.12);
  color: #f1edff;
  font-size: 13px;
  font-weight: 700;
  margin-bottom: 12px;
}

.rc-header h1 {
  font-size: 36px;
  line-height: 1.1;
  margin: 0 0 10px;
  color: #ffffff;
}

.rc-subtitle {
  color: #f3f0ff;
  font-size: 16px;
  line-height: 1.5;
  max-width: 780px;
  margin: 0;
}

.rc-header-badges {
  display: grid;
  gap: 8px;
  min-width: 230px;
}

.rc-badge {
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

.rc-badge span {
  color: #ddd6ff;
}

.rc-panel {
  border: 1px solid var(--rc-line) !important;
  border-radius: 8px !important;
  background: var(--rc-surface) !important;
  box-shadow: var(--rc-shadow);
  padding: 16px !important;
}

.rc-card-title {
  margin: 0 0 6px;
  font-size: 18px;
  line-height: 1.25;
  color: var(--rc-ink);
}

.rc-card-copy {
  margin: 0 0 12px;
  color: var(--rc-muted);
  font-size: 14px;
  line-height: 1.45;
}

.rc-alert {
  border-left: 4px solid var(--rc-warn);
  background: var(--rc-warn-soft);
  padding: 12px 14px;
  margin: 0 0 12px;
  border-radius: 6px;
  color: #4a2d00;
  font-size: 14px;
  line-height: 1.45;
}

.rc-side-note {
  border: 1px solid #d8dff0;
  border-radius: 8px;
  background: #f9faff;
  padding: 14px;
  color: var(--rc-muted);
  font-size: 14px;
  line-height: 1.45;
}

.rc-side-note strong {
  display: block;
  margin-bottom: 6px;
  color: var(--rc-ink);
}

.rc-actions button {
  min-height: 44px !important;
  border-radius: 6px !important;
  font-weight: 700 !important;
}

button.primary {
  background: var(--rc-accent) !important;
  border-color: var(--rc-accent) !important;
}

button.primary:hover {
  background: var(--rc-accent-dark) !important;
  border-color: var(--rc-accent-dark) !important;
}

details {
  border: 1px solid var(--rc-line);
  border-radius: 8px;
  padding: 10px 12px;
  background: #fff;
}

summary {
  cursor: pointer;
}

.rc-case-section {
  margin-bottom: 16px;
}

.rc-section-badge {
  display: inline-flex;
  width: fit-content;
  padding: 5px 10px;
  border: 1px solid #d7ccff;
  border-radius: 999px;
  background: #eee9ff;
  color: var(--rc-accent-dark);
  font-size: 12px;
  font-weight: 800;
  margin-bottom: 10px;
}

.rc-case-section h2 {
  margin: 0 0 8px;
  color: var(--rc-ink);
  font-size: 24px;
  line-height: 1.18;
}

.rc-case-section p {
  margin: 0 0 12px;
  color: var(--rc-muted);
  line-height: 1.55;
}

.rc-stat-grid,
.rc-method-grid {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 12px;
  margin-top: 16px;
}

.rc-method-grid {
  grid-template-columns: repeat(3, minmax(0, 1fr));
}

.rc-stat,
.rc-method-card {
  border: 1px solid var(--rc-line);
  border-radius: 8px;
  background: #f9faff;
  padding: 14px;
}

.rc-stat-num {
  display: block;
  color: var(--rc-accent-dark);
  font-size: 25px;
  font-weight: 800;
  line-height: 1;
  margin-bottom: 6px;
}

.rc-stat-label,
.rc-method-card span {
  display: block;
  color: var(--rc-muted);
  font-size: 13px;
  line-height: 1.4;
}

.rc-method-card strong {
  display: block;
  margin-bottom: 6px;
  color: var(--rc-ink);
}

.rc-pipeline {
  display: grid;
  gap: 12px;
  margin-top: 14px;
}

.rc-pipeline-step {
  display: grid;
  grid-template-columns: 110px minmax(0, 1fr);
  gap: 14px;
  align-items: start;
  border: 1px solid var(--rc-line);
  border-radius: 8px;
  background: #fbfcff;
  padding: 14px;
}

.rc-step-label {
  display: inline-flex;
  justify-content: center;
  align-items: center;
  min-height: 34px;
  border-radius: 6px;
  background: #eee9ff;
  color: var(--rc-accent-dark);
  font-size: 12px;
  font-weight: 800;
  text-transform: uppercase;
}

.rc-pipeline-step h4 {
  margin: 0 0 5px;
  color: var(--rc-ink);
  font-size: 15px;
}

.rc-pipeline-step p {
  margin: 0;
  color: var(--rc-muted);
  font-size: 14px;
  line-height: 1.45;
}

@media (max-width: 760px) {
  .rc-header {
    padding: 20px;
  }

  .rc-header h1 {
    font-size: 28px;
  }

  .rc-header-grid,
  .rc-pipeline-step {
    grid-template-columns: 1fr;
  }

  .rc-header-badges {
    min-width: 0;
  }

  .rc-stat-grid,
  .rc-method-grid {
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


def _document_label(path: Path) -> str:
    labels = {
        "a_redacao_no_enem_2025_cartilha_do_participante.pdf": "Cartilha do Participante 2025",
        "Situacoes_nota_zero.pdf": "Manual - Situacoes de nota zero",
        "Competencia_1.pdf": "Manual - Competencia 1",
        "Competencia_2.pdf": "Manual - Competencia 2",
        "Competencia_3.pdf": "Manual - Competencia 3",
        "Competencia_4.pdf": "Manual - Competencia 4",
        "Competencia_5.pdf": "Manual - Competencia 5",
    }
    return labels.get(path.name, path.stem)


def _classify_category(text: str, source: str) -> str:
    source_lower = source.lower()
    if "competencia_1" in source_lower:
        return "competencia_1"
    if "competencia_2" in source_lower:
        return "competencia_2"
    if "competencia_3" in source_lower:
        return "competencia_3"
    if "competencia_4" in source_lower:
        return "competencia_4"
    if "competencia_5" in source_lower:
        return "competencia_5"
    if "nota_zero" in source_lower or "situacoes" in source_lower:
        return "nota_zero"

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
        label = _document_label(pdf_path)
        for doc in loaded_docs:
            doc.metadata["documento"] = label
            doc.metadata["source"] = pdf_path.name
        documents.extend(loaded_docs)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=850,
        chunk_overlap=180,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)

    for chunk in chunks:
        chunk.metadata["categoria"] = _classify_category(
            chunk.page_content,
            chunk.metadata.get("source", ""),
        )

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
        document = doc.metadata.get("documento", "Documento")
        category = doc.metadata.get("categoria", "geral")
        excerpt = doc.page_content.replace("\n", " ").strip()[:800]
        sources.append(
            f"<details><summary><strong>Trecho {i}</strong> | "
            f"<code>{document}</code> | <code>{category}</code> | "
            f"Fonte: <code>{source}</code> | Pagina: <code>{page}</code></summary>\n\n"
            f"{excerpt}\n\n</details>"
        )
    return "\n\n".join(sources)


def answer_question(api_key: str, question: str):
    api_key = (api_key or "").strip()
    question = (question or "").strip()

    if not api_key:
        yield _format_status("Cole sua GROQ_API_KEY para testar o app.")
        return
    if not question:
        yield _format_status("Digite uma pergunta sobre a redacao do Enem.")
        return
    if len(question) > MAX_QUESTION_CHARS:
        yield _format_status(f"Sua pergunta esta muito longa. Use ate {MAX_QUESTION_CHARS} caracteres.")
        return

    cache_key = (_cache_key(api_key), question.lower())
    if cache_key in ANSWER_CACHE:
        yield ANSWER_CACHE[cache_key]
        return

    yield _format_status("Preparando a base de conhecimento da redacao do Enem...")

    try:
        vectorstore = _create_vectorstore()

        saved_chunks = vectorstore._collection.count()
        if saved_chunks > 0:
            yield _format_status(f"Banco vetorial carregado com {saved_chunks} trechos salvos. Buscando contexto...")
        else:
            yield _format_status(
                "Primeiro uso detectado: criando embeddings locais das cartilhas e manuais. "
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
            yield _format_status("Nao encontrei trechos relevantes nos documentos carregados.")
            return

        context = "\n\n".join(
            f"[{doc.metadata.get('documento', 'Documento')} | "
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
            f"## Resposta\n\n{answer}\n\n"
            "---\n\n"
            "### Fontes usadas\n\n"
            f"{_format_sources(docs)}\n\n"
            "---\n\n"
            "_Resposta gerada com base nos trechos recuperados dos documentos oficiais do Inep. "
            "Use como apoio de estudo; nao substitui professor, corretor ou a leitura integral da cartilha._"
        )
        ANSWER_CACHE[cache_key] = final_response
        yield final_response

    except Exception as error:
        yield _format_status(_friendly_error(error))


with gr.Blocks(title=APP_TITLE, css=CUSTOM_CSS) as demo:
    gr.HTML(
        """
        <header class="rc-header">
          <div class="rc-header-grid">
            <div>
              <div class="rc-kicker">RAG aplicado a criterios de redacao</div>
              <h1>Redacao Clara</h1>
              <p class="rc-subtitle">
                Entenda as competencias, criterios de avaliacao e situacoes de nota zero
                da redacao do Enem com base em documentos oficiais do Inep.
              </p>
            </div>
            <div class="rc-header-badges" aria-label="Resumo tecnico do projeto">
              <div class="rc-badge"><span>Fonte</span><strong>Inep</strong></div>
              <div class="rc-badge"><span>Busca</span><strong>ChromaDB</strong></div>
              <div class="rc-badge"><span>Embeddings</span><strong>Locais</strong></div>
            </div>
          </div>
        </header>
        """
    )

    with gr.Tabs():
        with gr.Tab("Assistente"):
            with gr.Row():
                with gr.Column(scale=7, elem_classes=["rc-panel"]):
                    gr.HTML(
                        """
                        <h2 class="rc-card-title">Consulta</h2>
                        <p class="rc-card-copy">
                          Pergunte sobre competencias, nota zero, proposta de intervencao,
                          fuga ao tema e criterios de avaliacao.
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
                        placeholder="Ex.: O que pode zerar a redacao do Enem?",
                        lines=4,
                        max_lines=6,
                        info="O app explica criterios; ele nao atribui nota oficial para redacoes.",
                    )

                    with gr.Row(elem_classes=["rc-actions"]):
                        button = gr.Button("Perguntar", variant="primary")
                        clear = gr.ClearButton([question], value="Limpar pergunta")

                with gr.Column(scale=4, elem_classes=["rc-panel"]):
                    gr.HTML(
                        """
                        <div class="rc-alert">
                          <strong>Uso seguro</strong><br>
                          Este app e um apoio de estudo. Ele nao substitui professor, corretor,
                          simulado de redacao ou a leitura completa da cartilha oficial.
                        </div>
                        <div class="rc-side-note">
                          <strong>Base carregada</strong>
                          Cartilha do Participante 2025, manual de nota zero e manuais das
                          competencias 1 a 5.
                        </div>
                        """
                    )

            gr.Examples(
                examples=[
                    "Explique as 5 competencias da redacao do Enem em linguagem simples.",
                    "O que pode zerar a redacao do Enem?",
                    "O que a competencia 5 exige na proposta de intervencao?",
                    "Qual a diferenca entre fuga ao tema e tangenciamento?",
                    "Como a competencia 4 avalia coesao?",
                ],
                inputs=question,
                label="Perguntas de teste",
            )

            output = gr.Markdown(label="Resposta")

            button.click(
                fn=answer_question,
                inputs=[api_key, question],
                outputs=output,
            )

        with gr.Tab("Sobre o projeto"):
            gr.HTML(
                """
                <section class="rc-panel rc-case-section">
                  <div class="rc-section-badge">Fase 1</div>
                  <h2>Problema e proposta</h2>
                  <p>
                    A redacao do Enem tem criterios bem definidos, mas muitos estudantes
                    ainda estudam por dicas soltas. O Redacao Clara transforma cartilhas
                    oficiais em uma consulta guiada, com trechos usados como fonte.
                  </p>
                  <div class="rc-stat-grid">
                    <div class="rc-stat">
                      <span class="rc-stat-num">7</span>
                      <span class="rc-stat-label">documentos oficiais carregados</span>
                    </div>
                    <div class="rc-stat">
                      <span class="rc-stat-num">5</span>
                      <span class="rc-stat-label">competencias da matriz de redacao</span>
                    </div>
                    <div class="rc-stat">
                      <span class="rc-stat-num">0</span>
                      <span class="rc-stat-label">tokens gastos para embeddings em API</span>
                    </div>
                    <div class="rc-stat">
                      <span class="rc-stat-num">100%</span>
                      <span class="rc-stat-label">das respostas exibem fontes recuperadas</span>
                    </div>
                  </div>
                </section>
                """
            )

            gr.HTML(
                """
                <section class="rc-panel rc-case-section">
                  <div class="rc-section-badge">Fase 2</div>
                  <h2>Pipeline RAG</h2>
                  <p>
                    O sistema primeiro busca trechos nos documentos oficiais. Depois a LLM
                    gera a resposta somente com o contexto recuperado.
                  </p>
                  <div class="rc-pipeline">
                    <div class="rc-pipeline-step">
                      <span class="rc-step-label">Entrada</span>
                      <div>
                        <h4>Cartilhas e manuais em PDF</h4>
                        <p>Os documentos do Inep sao carregados com PyPDFLoader e recebem metadados.</p>
                      </div>
                    </div>
                    <div class="rc-pipeline-step">
                      <span class="rc-step-label">Texto</span>
                      <div>
                        <h4>Chunks com sobreposicao</h4>
                        <p>O conteudo e dividido em unidades menores para preservar contexto e melhorar a busca.</p>
                      </div>
                    </div>
                    <div class="rc-pipeline-step">
                      <span class="rc-step-label">Vetores</span>
                      <div>
                        <h4>Embeddings locais</h4>
                        <p>Sentence Transformers cria representacoes semanticas sem consumir quota da Groq.</p>
                      </div>
                    </div>
                    <div class="rc-pipeline-step">
                      <span class="rc-step-label">Busca</span>
                      <div>
                        <h4>ChromaDB e retriever</h4>
                        <p>Os vetores ficam persistidos e o retriever retorna os trechos mais relevantes.</p>
                      </div>
                    </div>
                    <div class="rc-pipeline-step">
                      <span class="rc-step-label">Resposta</span>
                      <div>
                        <h4>Groq com prompt de estudo</h4>
                        <p>A resposta explica criterios em linguagem simples e mostra fontes usadas.</p>
                      </div>
                    </div>
                  </div>
                </section>
                """
            )

            gr.HTML(
                """
                <section class="rc-panel rc-case-section">
                  <div class="rc-section-badge">Fase 3</div>
                  <h2>Ferramentas e metodos</h2>
                  <p>
                    O projeto reaproveita a arquitetura do Bula Clara em um novo dominio:
                    regras oficiais de avaliacao textual.
                  </p>
                  <div class="rc-method-grid">
                    <div class="rc-method-card">
                      <strong>Gradio</strong>
                      <span>Interface web com exemplos, resposta e fontes.</span>
                    </div>
                    <div class="rc-method-card">
                      <strong>LangChain</strong>
                      <span>Loader, splitter, prompt e retriever em um fluxo RAG.</span>
                    </div>
                    <div class="rc-method-card">
                      <strong>ChromaDB</strong>
                      <span>Persistencia local dos embeddings para reuso.</span>
                    </div>
                    <div class="rc-method-card">
                      <strong>Sentence Transformers</strong>
                      <span>Embeddings locais em portugues, sem billing de embeddings.</span>
                    </div>
                    <div class="rc-method-card">
                      <strong>Groq</strong>
                      <span>LLM em cloud para gerar explicacoes curtas e praticas.</span>
                    </div>
                    <div class="rc-method-card">
                      <strong>Metadados</strong>
                      <span>Documento, pagina e categoria para rastrear as respostas.</span>
                    </div>
                  </div>
                </section>
                """
            )


if __name__ == "__main__":
    demo.queue(default_concurrency_limit=2)
    demo.launch()
