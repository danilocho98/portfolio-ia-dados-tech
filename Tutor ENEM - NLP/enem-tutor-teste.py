import pandas as pd
import json
import os
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Garante stopwords baixadas
try:
    from nltk.corpus import stopwords
except ImportError:
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "nltk"])
    from nltk.corpus import stopwords

try:
    stopwords.words('portuguese')
except LookupError:
    nltk.download('stopwords', quiet=True)
    from nltk.corpus import stopwords

# 1. Carrega o banco de dados de questões do ENEM (Cache Local)
local_questions = "enem_questoes.json"
if not os.path.exists(local_questions):
    print("Erro: enem_questoes.json não encontrado. Rode o script de teste anterior primeiro.")
    exit(1)

df_questions = pd.read_json(local_questions)
df_questions.dropna(inplace=True)
df_questions = df_questions[df_questions['area'] != 'UNDEFINED']
df_questions.reset_index(drop=True, inplace=True)

# 2. Carrega a base local de matérias de estudo do Brasil Escola
local_materials = "materias_estudo.json"
if not os.path.exists(local_materials):
    print("Erro: materias_estudo.json não encontrado.")
    exit(1)

with open(local_materials, "r", encoding="utf-8") as f:
    materials_db = json.load(f)

# 3. Treina o Vetorizador TF-IDF com as questões do ENEM
tfidf = TfidfVectorizer(
    min_df=2, 
    encoding='utf-8', 
    ngram_range=(1, 2),
    stop_words=stopwords.words('portuguese')
)
features_questions = tfidf.fit_transform(df_questions['enunciado'].values.astype('U'))

# 4. Cria os vetores das palavras-chave das matérias de estudo
material_keys = list(materials_db.keys())
features_materials = tfidf.transform(material_keys)

def tutor_enem_analisar(duvida_usuario):
    """
    Simula o tutor de estudos completo:
    - Identifica similaridade de questões
    - Classifica a área mais provável
    - Recomenda matérias de estudo mapeadas do Brasil Escola
    """
    # Vetoriza a dúvida do usuário
    vetor_duvida = tfidf.transform([duvida_usuario])
    
    # A. Busca questões similares do ENEM
    similaridades_questoes = cosine_similarity(vetor_duvida, features_questions).flatten()
    top_indices_questoes = similaridades_questoes.argsort()[-2:][::-1] # Pega as 2 mais próximas
    
    questoes_sugeridas = []
    for idx in top_indices_questoes:
        questoes_sugeridas.append({
            "enunciado": df_questions.iloc[idx]['enunciado'],
            "area": df_questions.iloc[idx]['area'],
            "similaridade": similaridades_questoes[idx]
        })
    
    # B. Busca materiais de estudo parecidos
    similaridades_materias = cosine_similarity(vetor_duvida, features_materials).flatten()
    best_m_idx = similaridades_materias.argmax()
    best_m_score = similaridades_materias[best_m_idx]
    
    materias_recomendadas = []
    # Usamos uma nota de corte de similaridade (ex: 0.10) para sugerir links relevantes
    if best_m_score > 0.08:
        chave_encontrada = material_keys[best_m_idx]
        materia_info = materials_db[chave_encontrada]
        materias_recomendadas.append({
            "titulo": materia_info["titulo"],
            "link": materia_info["link"],
            "area": materia_info["area"],
            "chave_de_busca": chave_encontrada,
            "similaridade": best_m_score
        })
        
    return questoes_sugeridas, materias_recomendadas

# Testando o funcionamento
if __name__ == "__main__":
    print("="*60)
    print(" SIMULADOR DE TUTORIA INTELIGENTE — ENEM ")
    print("="*60)
    
    testes = [
        "Qual a relação entre o feudalismo e a suserania e vassalagem na Idade Média?",
        "Qual a fórmula de cálculo de corrente elétrica e resistores?"
    ]
    
    for query in testes:
        print(f"\nDÚVIDA DO ESTUDANTE: '{query}'")
        questoes, materias = tutor_enem_analisar(query)
        
        # Exibe matérias de estudo recomendadas
        if materias:
            print("\n=== MATERIAIS DE ESTUDO RECOMENDADOS (BRASIL ESCOLA) ===")
            for m in materias:
                print(f" -> Artigo: '{m['titulo']}'")
                print(f"    Link: {m['link']}")
                print(f"    Área temática: {m['area'].upper()} (Match: {m['similaridade']:.4f})")
        else:
            print("\n=== NENHUM MATERIAL DE ESTUDO ESPECÍFICO ENCONTRADO NO BANCO LOCAL ===")
            
        # Exibe questões parecidas do ENEM
        print("\n=== QUESTÕES SIMILARES DO ENEM ===")
        for i, q in enumerate(questoes, 1):
            resumo_enunciado = q['enunciado'][:220].replace('\n', ' ') + "..."
            print(f"   [{i}] {resumo_enunciado}")
            print(f"       Área: {q['area'].upper()} | Similaridade: {q['similaridade']:.4f}")
        print("-" * 60)
