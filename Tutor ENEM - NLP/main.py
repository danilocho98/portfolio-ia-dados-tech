import os
import sys
import subprocess
import urllib.parse
import re

# 1. Garante que todas as dependências do FastAPI e NLP estão instaladas
REQUIRED_PACKAGES = ["fastapi", "uvicorn", "pandas", "scikit-learn", "nltk"]
for package in REQUIRED_PACKAGES:
    try:
        if package == "fastapi":
            import fastapi
        elif package == "uvicorn":
            import uvicorn
        elif package == "pandas":
            import pandas
        elif package == "scikit-learn":
            import sklearn
        elif package == "nltk":
            import nltk
    except ImportError:
        print(f"Instalando {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import pandas as pd
import json
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Baixa as stopwords se necessário
try:
    from nltk.corpus import stopwords
    stopwords.words('portuguese')
except (ImportError, LookupError):
    import nltk
    nltk.download('stopwords', quiet=True)
    from nltk.corpus import stopwords

app = FastAPI(title="Tutor ENEM NLP API", version="1.0")

# 2. Carrega as bases de dados locais
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
QUESTIONS_FILE = os.path.join(BASE_DIR, "enem_questoes.json")
MATERIALS_FILE = os.path.join(BASE_DIR, "materias_estudo.json")

if not os.path.exists(QUESTIONS_FILE) or not os.path.exists(MATERIALS_FILE):
    print("ERRO: Banco de dados local não encontrado. Certifique-se de que 'enem_questoes.json' e 'materias_estudo.json' existem.")
    sys.exit(1)

# Inicializa as variáveis do modelo globalmente para alta performance
df_questions = None
materials_db = None
tfidf = None
features_questions = None
features_materials = None
material_keys = None

@app.on_event("startup")
def startup_event():
    """
    Treina o vetorizador TF-IDF e carrega as bases na inicialização do servidor
    para que as requisições à API sejam instantâneas.
    """
    global df_questions, materials_db, tfidf, features_questions, features_materials, material_keys
    
    # Carrega questões
    df_questions = pd.read_json(QUESTIONS_FILE)
    df_questions.dropna(inplace=True)
    df_questions = df_questions[df_questions['area'] != 'UNDEFINED']
    # Limpa caracteres corrompidos (\ufffd) para melhorar legibilidade
    df_questions['enunciado'] = df_questions['enunciado'].str.replace('\ufffd', '', regex=False)
    df_questions.reset_index(drop=True, inplace=True)
    
    # Carrega matérias de estudo
    with open(MATERIALS_FILE, "r", encoding="utf-8") as f:
        materials_db = json.load(f)
        
    # Inicializa vetorizador
    tfidf = TfidfVectorizer(
        min_df=2, 
        encoding='utf-8', 
        ngram_range=(1, 2),
        stop_words=stopwords.words('portuguese')
    )
    
    # Vetoriza banco de dados
    features_questions = tfidf.fit_transform(df_questions['enunciado'].values.astype('U'))
    
    # Vetoriza palavras-chave dos materiais de estudo
    material_keys = list(materials_db.keys())
    features_materials = tfidf.transform(material_keys)
    print("Servidor FastAPI iniciado. Modelo TF-IDF carregado com sucesso.")

def limpar_e_codificar_query(query: str):
    # Remove acentos/caracteres especiais simples e converte para minúsculas
    query_clean = query.lower()
    # Remove pontuações e caracteres especiais
    query_clean = re.sub(r'[^\w\s]', ' ', query_clean)
    words = query_clean.split()
    
    # Filtra as stopwords de português se NLTK stopwords estiver disponível
    from nltk.corpus import stopwords
    try:
        stops = set(stopwords.words('portuguese'))
    except Exception:
        stops = set()
        
    filtered_words = [w for w in words if w not in stops and len(w) > 1]
    
    # Se todas as palavras forem stopwords, usa a query original limpa
    if not filtered_words:
        filtered_words = words
        
    clean_text = " ".join(filtered_words)
    encoded_text = urllib.parse.quote_plus(clean_text)
    return clean_text, encoded_text

class AnalyzeRequest(BaseModel):
    query: str

@app.post("/api/analyze")
def analyze_query(request: AnalyzeRequest):
    """
    Rota da API que analisa a dúvida do usuário, prediz a área mais provável do ENEM,
    sugere questões similares e indica artigos de estudo do Brasil Escola.
    """
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="A consulta não pode estar vazia.")
        
    # Transforma a dúvida do usuário no vetor TF-IDF
    vetor_duvida = tfidf.transform([request.query])
    
    # 1. Classificação de Área mais provável (usando a similaridade com o corpus)
    similaridades_questoes = cosine_similarity(vetor_duvida, features_questions).flatten()
    
    # Se todas as similaridades forem 0 (nenhuma palavra bateu), retorna uma resposta neutra
    if similaridades_questoes.max() == 0:
        return {
            "area_predita": "Indeterminada (Sem palavras conhecidas)",
            "questoes_similares": [],
            "materias_estudo": []
        }
    
    # Identifica os índices das 2 questões mais similares do ENEM
    top_indices_questoes = similaridades_questoes.argsort()[-2:][::-1]
    
    questoes_sugeridas = []
    areas_encontradas = []
    
    for idx in top_indices_questoes:
        area = df_questions.iloc[idx]['area']
        areas_encontradas.append(area)
        questoes_sugeridas.append({
            "enunciado": df_questions.iloc[idx]['enunciado'],
            "area": area,
            "similaridade": float(similaridades_questoes[idx])
        })
        
    # Define a área predita como a da questão mais similar (vizinho mais próximo no espaço TF-IDF)
    area_predita = questoes_sugeridas[0]["area"] if questoes_sugeridas else "Indeterminada"

    # 2. Busca materiais de estudo parecidos
    similaridades_materias = cosine_similarity(vetor_duvida, features_materials).flatten()
    best_m_idx = similaridades_materias.argmax()
    best_m_score = float(similaridades_materias[best_m_idx])
    
    materias_recomendadas = []
    
    # Se encontramos um casamento específico acima do limiar (0.05)
    if best_m_score > 0.05:
        chave_encontrada = material_keys[best_m_idx]
        materia_info = materials_db[chave_encontrada]
        materias_recomendadas.append({
            "titulo": materia_info["titulo"],
            "link": materia_info["link"],
            "area": materia_info["area"],
            "similaridade": best_m_score,
            "tipo": "especifica"
        })
    else:
        # Fallback: Gera um link de busca interna direta no Brasil Escola
        texto_limpo, query_codificada = limpar_e_codificar_query(request.query)
        if texto_limpo:
            titulo_busca = f"Pesquisar sobre: {texto_limpo}"
            link_busca = f"https://brasilescola.uol.com.br/busca?q={query_codificada}"
        else:
            titulo_busca = "Pesquisar Dúvida no Brasil Escola"
            link_busca = f"https://brasilescola.uol.com.br/busca?q={urllib.parse.quote_plus(request.query)}"
            
        materias_recomendadas.append({
            "titulo": titulo_busca,
            "link": link_busca,
            "area": area_predita,
            "similaridade": 0.0,
            "tipo": "busca"
        })
        
    return {
        "area_predita": area_predita,
        "questoes_similares": questoes_sugeridas,
        "materias_estudo": materias_recomendadas
    }

# 3. Serve os arquivos estáticos do frontend (CSS, JS) e o index.html principal
static_path = os.path.join(BASE_DIR, "static")

# Mecanismo de Fallback: se index.html estiver solto na raiz e não dentro da pasta static,
# redirecionamos o diretório estático para a raiz para evitar falha no carregamento.
if not os.path.exists(os.path.join(static_path, "index.html")) and os.path.exists(os.path.join(BASE_DIR, "index.html")):
    print("AVISO: index.html detectado na raiz. Configurando diretório estático de fallback para a raiz.")
    static_path = BASE_DIR
elif not os.path.exists(static_path):
    os.makedirs(static_path)

app.mount("/static", StaticFiles(directory=static_path), name="static")

@app.get("/", response_class=HTMLResponse)
def read_index():
    index_file = os.path.join(static_path, "index.html")
    if not os.path.exists(index_file):
        # Tenta buscar na raiz
        fallback_file = os.path.join(BASE_DIR, "index.html")
        if os.path.exists(fallback_file):
            with open(fallback_file, "r", encoding="utf-8") as f:
                return HTMLResponse(content=f.read(), status_code=200)
        return "<h3>index.html não encontrado na pasta static/ nem na raiz do projeto</h3>"
    with open(index_file, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read(), status_code=200)

if __name__ == "__main__":
    # Inicia o servidor uvicorn na porta 8000
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
