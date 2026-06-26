document.addEventListener("DOMContentLoaded", () => {
    // ── NAVEGAÇÃO DE ABAS ──
    const tabButtons = document.querySelectorAll(".tab-btn");
    const tabContents = document.querySelectorAll(".tab-content");

    tabButtons.forEach(button => {
        button.addEventListener("click", () => {
            // Remove active de todos os botões e abas
            tabButtons.forEach(btn => btn.classList.remove("active"));
            tabContents.forEach(content => content.classList.remove("active"));

            // Adiciona active no botão clicado e na aba correspondente
            button.classList.add("active");
            const tabId = button.getAttribute("data-tab");
            document.getElementById(`tab-${tabId}`).classList.add("active");
        });
    });

    // ── SUBMIT DO FORMULÁRIO DO TUTOR ──
    const tutorForm = document.getElementById("tutor-form");
    const queryInput = document.getElementById("query-input");
    const submitBtn = document.getElementById("submit-btn");
    const btnText = submitBtn.querySelector(".btn-text");
    const spinner = submitBtn.querySelector(".spinner");
    const resultsPanel = document.getElementById("results-panel");
    const predictedBadge = document.getElementById("predicted-area-badge");
    const materialContent = document.getElementById("material-content");
    const questionsList = document.getElementById("questions-list");

    tutorForm.addEventListener("submit", async (e) => {
        e.preventDefault();

        const queryText = queryInput.value.trim();
        if (!queryText) return;

        // Ativa o estado de carregamento
        submitBtn.disabled = true;
        btnText.textContent = "Analisando...";
        spinner.classList.remove("hidden");
        resultsPanel.classList.add("hidden");

        try {
            const response = await fetch("/api/analyze", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify({ query: queryText })
            });

            if (!response.ok) {
                throw new Error("Falha na análise do servidor.");
            }

            const data = await response.json();
            renderResults(data);

        } catch (error) {
            console.error("Erro:", error);
            alert("Ocorreu um erro ao processar a sua pergunta. Tente novamente.");
        } finally {
            // Desativa o estado de carregamento
            submitBtn.disabled = false;
            btnText.textContent = "Analisar Dúvida";
            spinner.classList.add("hidden");
        }
    });

    // ── RENDERIZAÇÃO DOS RESULTADOS ──
    function renderResults(data) {
        // 1. Renderiza a Área Prevista
        const area = data.area_predita.toLowerCase();
        predictedBadge.textContent = data.area_predita;
        
        // Remove classes de cor antigas e adiciona a nova
        predictedBadge.className = "badge";
        if (area.includes("natureza")) {
            predictedBadge.classList.add("natureza");
        } else if (area.includes("humanas")) {
            predictedBadge.classList.add("humanas");
        } else if (area.includes("linguagens")) {
            predictedBadge.classList.add("linguagens");
        } else if (area.includes("matemática")) {
            predictedBadge.classList.add("matematica");
        }

        // 2. Renderiza Materiais de Estudo Recomendados
        materialContent.innerHTML = "";
        if (data.materias_estudo && data.materias_estudo.length > 0) {
            data.materias_estudo.forEach(mat => {
                const matchText = mat.tipo === "busca"
                    ? `Nenhum atalho mapeado na base local. Clique para pesquisar o tema diretamente no buscador do Brasil Escola.`
                    : `Matéria associada à palavra-chave (Match: ${(mat.similaridade * 100).toFixed(1)}%)`;
                const materialHTML = `
                    <div class="study-material-rec">
                        <h4>${mat.titulo}</h4>
                        <p style="font-size: 8.5pt; color: var(--text-muted); margin-bottom: 8px;">${matchText}</p>
                        <a href="${mat.link}" target="_blank" class="visit-link">Pesquisar Artigos no Portal</a>
                    </div>
                `;
                materialContent.innerHTML = materialHTML;
            });
        } else {
            materialContent.innerHTML = `
                <p style="font-size: 9pt; color: var(--text-muted); font-style: italic;">
                    Nenhum material encontrado para esse termo no banco de estudos local.
                </p>
            `;
        }

        // 3. Renderiza Questões Similares (Accordion)
        questionsList.innerHTML = "";
        if (data.questoes_similares && data.questoes_similares.length > 0) {
            data.questoes_similares.forEach((q, index) => {
                // Cria resumo do enunciado para o título do accordion
                let resumo = q.enunciado.replace(/\\n/g, ' ').replace(/\n/g, ' ');
                if (resumo.length > 80) resumo = resumo.substring(0, 80) + "...";

                const questionHTML = `
                    <div class="accordion-item">
                        <button class="accordion-header" type="button">
                            <span><strong>Questão ${index + 1}</strong>: ${resumo} (${(q.similaridade * 100).toFixed(1)}% similar)</span>
                            <span class="accordion-indicator">▼</span>
                        </button>
                        <div class="accordion-body">
                            <div class="accordion-content">
                                <p style="margin-bottom: 8px;"><strong>Área do ENEM:</strong> <span style="text-transform: uppercase;">${q.area}</span></p>
                                <p>${q.enunciado.replace(/\\n/g, '<br/>').replace(/\n/g, '<br/>')}</p>
                            </div>
                        </div>
                    </div>
                `;
                questionsList.insertAdjacentHTML("beforeend", questionHTML);
            });

            // Ativa os listeners de click do Accordion
            initAccordion();

        } else {
            questionsList.innerHTML = `
                <p style="font-size: 9pt; color: var(--text-muted); font-style: italic;">
                    Nenhuma questão similar encontrada.
                </p>
            `;
        }

        // Exibe o painel de resultados com animação
        resultsPanel.classList.remove("hidden");
        resultsPanel.scrollIntoView({ behavior: "smooth", block: "nearest" });
    }

    // ── LÓGICA DO ACCORDION (EXPAND/COLLAPSE) ──
    function initAccordion() {
        const accordionHeaders = document.querySelectorAll(".accordion-header");
        
        accordionHeaders.forEach(header => {
            header.addEventListener("click", () => {
                const item = header.parentElement;
                
                // Fecha outros accordions abertos (opcional, para comportamento sanfona clássico)
                const openItems = document.querySelectorAll(".accordion-item.open");
                openItems.forEach(openItem => {
                    if (openItem !== item) {
                        openItem.classList.remove("open");
                    }
                });

                // Alterna o estado do item clicado
                item.classList.toggle("open");
            });
        });
    }
});
