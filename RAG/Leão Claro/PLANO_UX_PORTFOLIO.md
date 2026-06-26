# Plano de UX e Apresentacao para Projetos de Portfolio

Este documento registra melhorias reutilizaveis para projetos de IA, RAG, dashboards e apps publicados como portfolio.

O objetivo nao e apenas deixar a interface bonita. A ideia e comunicar maturidade: clareza de uso, performance percebida, confianca, explicacao tecnica e valor real do projeto.

## Contexto do Projeto

Projeto atual: **Leao Claro**

Descricao curta:

> Assistente RAG que responde duvidas sobre IRPF com base em documentos oficiais da Receita Federal, usando LangChain, Gemini e ChromaDB para recuperar trechos relevantes e gerar respostas contextualizadas com fontes.

Importancia:

- Ajuda pessoas a entenderem documentos longos e tecnicos.
- Reduz dependencia de respostas soltas sem fonte.
- Demonstra uso responsavel de IA em um dominio sensivel.
- Mostra conhecimento em RAG, embeddings, banco vetorial, LLMs e design de produto.

Tecnologias e metodos usados:

- Python
- Gradio
- LangChain
- Gemini API
- ChromaDB
- Embeddings
- PDF document loading
- Retrieval-Augmented Generation
- Prompt engineering
- Persistencia local de base vetorial
- Uso de chave de API fornecida pelo proprio usuario

## 1. Skeleton Loaders

Ideia original:

> Trocar loading spinner por barras cinzas/placeholder que indicam a estrutura da pagina antes dos dados carregarem.

Aplicabilidade neste projeto: **media**

Em apps React, Next.js, Streamlit customizado ou frontends proprios, skeleton loaders sao excelentes. Eles deixam o carregamento parecer mais rapido porque o usuario ja entende o layout antes do conteudo chegar.

No Leao Claro, como a interface atual usa Gradio, a customizacao visual e mais limitada. Ainda assim, da para aplicar a ideia com mensagens progressivas e blocos de resposta parciais.

Implementacao recomendada no Leao Claro:

- Mostrar mensagens de progresso em vez de uma tela muda.
- Exibir etapas como:
  - Preparando base de conhecimento
  - Carregando banco vetorial
  - Indexando documento oficial
  - Buscando trechos relevantes
  - Gerando resposta
- No futuro, se migrar para React, usar skeleton real para:
  - card de resposta
  - lista de fontes
  - area de contexto usado

Valor para portfolio:

- Demonstra preocupacao com performance percebida.
- Mostra que o projeto foi pensado como produto, nao apenas como notebook.

Prioridade: **media**

## 2. Caching

Ideia original:

> Se todo clique demora, o app parece lento. Cache permite carregar uma vez e reutilizar.

Aplicabilidade neste projeto: **alta**

Caching e uma das melhorias mais importantes para RAG. Sem cache, o app pode recriar embeddings ou recalcular dados toda vez. Isso custa tempo, tokens e quota de API.

O que ja existe no Leao Claro:

- O ChromaDB salva os embeddings em `chroma_leao_claro_irpf_gemini`.
- Se a pasta ja tiver dados, o app carrega a base em vez de recriar embeddings.

Melhorias futuras:

- Cachear respostas para perguntas repetidas.
- Cachear a inicializacao do vectorstore no app.
- Cachear o retriever.
- Evitar recriar objetos pesados a cada pergunta.
- Separar primeira indexacao de uso normal.

Exemplo de comportamento ideal:

- Primeiro uso:
  - cria embeddings
  - salva Chroma
  - demora mais
- Proximos usos:
  - carrega Chroma
  - busca contexto
  - responde rapidamente

Valor para portfolio:

- Mostra conhecimento de custo operacional.
- Mostra preocupacao com escalabilidade.
- Evita gasto desnecessario de API.

Prioridade: **alta**

## 3. Optimistic Rendering

Ideia original:

> Se uma acao quase sempre funciona, atualize a UI imediatamente e reconcilie depois com o servidor.

Aplicabilidade neste projeto: **baixa a media**

Optimistic rendering funciona muito bem para acoes como:

- curtir post
- marcar tarefa como concluida
- adicionar item numa lista
- salvar preferencia visual

No Leao Claro, a acao principal e gerar resposta com IA. Nao da para assumir que funcionou, porque a resposta depende de API, quota, chave valida, contexto e modelo.

O que faz sentido aplicar:

- Assim que o usuario clicar em "Perguntar", mostrar imediatamente:
  - pergunta recebida
  - status de processamento
  - area reservada para resposta
- Desabilitar temporariamente o botao para evitar duplo clique.
- Mostrar erro claro se falhar.

O que nao faz sentido:

- Exibir resposta final antes da API responder.
- Fingir sucesso em chamada de modelo.

Valor para portfolio:

- Mostra criterio tecnico: nem toda dica de UX se aplica a todo tipo de app.

Prioridade: **baixa**

## 4. Tooltips

Ideia original:

> Botoes apenas com icones devem ter mensagens ao passar o mouse.

Aplicabilidade neste projeto: **baixa no estado atual, alta se houver interface mais rica**

O Leao Claro atual usa poucos controles e praticamente nenhum botao icon-only. Portanto, tooltips nao sao urgentes.

Quando aplicar:

- Botao de limpar conversa
- Botao de copiar resposta
- Botao de mostrar/ocultar fontes
- Botao de baixar resposta
- Botao de atualizar base vetorial

Boas tooltips:

- "Copiar resposta"
- "Mostrar trechos usados como fonte"
- "Limpar pergunta e resposta"
- "Recriar base vetorial"

Valor para portfolio:

- Mostra cuidado com usabilidade sem poluir a interface.

Prioridade: **baixa agora, media em uma versao mais completa**

## 5. Transparencia de Fontes

Sugestao adicional.

Aplicabilidade neste projeto: **muito alta**

Em projetos RAG, a confianca vem das fontes. O usuario precisa saber de onde saiu a resposta.

O que implementar:

- Mostrar os trechos usados.
- Mostrar pagina do PDF.
- Separar resposta de contexto.
- Avisar quando nao houver base suficiente.

O que o Leao Claro ja faz:

- Mostra trechos recuperados.
- Mostra fonte e pagina.
- Pede ao modelo para responder somente com base no contexto.

Melhorias futuras:

- Botao "Ver fontes".
- Destaque visual para os trechos.
- Link ou indicacao clara do PDF oficial.

Valor para portfolio:

- Excelente. RAG sem fonte parece chatbot comum. RAG com fonte parece produto confiavel.

Prioridade: **alta**

## 6. Avisos de Escopo e Seguranca

Sugestao adicional.

Aplicabilidade neste projeto: **muito alta**

IRPF e tema sensivel. O app deve deixar claro o que ele faz e o que nao faz.

Avisos importantes:

- Nao substitui contador.
- Nao substitui a Receita Federal.
- Nao solicitar CPF.
- Nao solicitar senha gov.br.
- Nao solicitar dados bancarios.
- Nao prometer economia tributaria.
- Responder apenas com base no documento oficial.

Valor para portfolio:

- Mostra maturidade e responsabilidade no uso de IA.
- Diferencia o projeto de demos irresponsaveis.

Prioridade: **alta**

## 7. Pagina de Apresentacao do Projeto

Sugestao adicional para portfolio.

Aplicabilidade neste projeto: **alta**

Todo projeto de portfolio deveria responder rapidamente:

- O que e?
- Para quem serve?
- Qual dor resolve?
- Quais tecnologias usa?
- Como funciona?
- Quais limitacoes existem?
- Como rodar?
- O que eu aprendi/construi?

Estrutura recomendada no README:

- Nome do projeto
- Descricao curta
- Problema
- Solucao
- Tecnologias
- Arquitetura
- Como usar
- Como publicar
- Limitacoes
- Proximos passos

Valor para portfolio:

- Recrutadores e avaliadores entendem o projeto sem precisar abrir o codigo.

Prioridade: **alta**

## 8. Estados de Interface

Sugestao adicional.

Aplicabilidade neste projeto: **alta**

Um app serio deve tratar estados diferentes:

- vazio
- carregando
- sucesso
- erro
- chave ausente
- chave invalida
- quota excedida
- documento nao encontrado
- sem contexto suficiente

Melhorias recomendadas:

- Mensagens amigaveis para cada erro.
- Explicar o que o usuario pode fazer.
- Evitar stack trace cru na interface.

Exemplos:

- "Sua chave nao parece valida. Gere uma chave no Google AI Studio."
- "A quota gratuita foi atingida. Aguarde alguns minutos e tente novamente."
- "Nao encontrei essa informacao no documento oficial usado como fonte."

Valor para portfolio:

- Mostra que o app foi pensado para usuarios reais.

Prioridade: **alta**

## 9. Reducao de Custo e Quota

Sugestao adicional.

Aplicabilidade neste projeto: **muito alta**

Como o app usa API, custo e limite importam.

Estrategias:

- Persistir ChromaDB.
- Evitar recriar embeddings.
- Reduzir quantidade de chunks.
- Usar `k=3` ou `k=4` no retriever.
- Pedir chave do proprio usuario.
- Cachear perguntas frequentes.
- Mostrar aviso quando a primeira indexacao puder demorar.

Valor para portfolio:

- Demonstra conhecimento alem do prototipo.

Prioridade: **alta**

## 10. Sugestao de Roadmap

### Versao 1

- Notebook limpo e explicativo.
- App Gradio funcional.
- Usuario informa propria chave.
- Respostas com fontes.
- README completo.

### Versao 2

- Mensagens de progresso melhores.
- Cache mais robusto.
- Tratamento de erros amigavel.
- Botao para copiar resposta.
- Botao para mostrar/ocultar fontes.

### Versao 3

- Interface customizada em React ou Next.js.
- Skeleton loaders reais.
- Historico local de perguntas.
- Exportacao da resposta em Markdown/PDF.
- Pagina de arquitetura do projeto.

## Checklist Reutilizavel

Antes de publicar um projeto de IA no portfolio:

- [ ] O README explica a dor real?
- [ ] O README lista ferramentas e metodos usados?
- [ ] O app tem aviso de limitacao?
- [ ] O app evita coletar dados sensiveis?
- [ ] O app mostra fontes ou evidencias?
- [ ] O app trata erros comuns?
- [ ] O app informa quando esta carregando?
- [ ] O app evita custo desnecessario?
- [ ] O `.env` esta no `.gitignore`?
- [ ] Nenhuma chave de API foi commitada?
- [ ] Existe uma descricao curta para recrutadores?
- [ ] Existe uma explicacao tecnica para avaliadores?

## Observacao de Seguranca

Se uma chave de API aparecer em print, video, chat, commit ou tela compartilhada, trate como exposta.

Acao recomendada:

1. Revogar a chave antiga.
2. Criar uma chave nova.
3. Guardar somente em `.env` ou em Secrets da plataforma.
4. Nunca publicar `.env`.
