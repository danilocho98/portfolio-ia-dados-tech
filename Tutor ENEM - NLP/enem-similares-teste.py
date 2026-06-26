import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Certifica que os pacotes necessários estão instalados/baixados
try:
    import nltk
    from nltk.corpus import stopwords
except ImportError:
    print("Instalando nltk...")
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "nltk"])
    import nltk
    from nltk.corpus import stopwords

# Baixa as stopwords se necessário
try:
    stopwords.words('portuguese')
except LookupError:
    nltk.download('stopwords', quiet=True)
    from nltk.corpus import stopwords

# 1. Carrega os dados do ENEM (com cache local para evitar falhas de rede)
import os
import urllib.request

local_file = "enem_questoes.json"
url = "https://raw.githubusercontent.com/jaumpedro214/enem-nlp-classification/main/data/enem_questoes.json"

if not os.path.exists(local_file):
    print("Baixando banco de questões do ENEM (cache local)...")
    try:
        opener = urllib.request.build_opener()
        opener.addheaders = [('User-agent', 'Mozilla/5.0')]
        urllib.request.install_opener(opener)
        urllib.request.urlretrieve(url, local_file)
        print("Download concluído e salvo localmente em 'enem_questoes.json'.")
    except Exception as e:
        print(f"Falha ao baixar cache local: {e}. Tentando ler diretamente da URL...")
        local_file = url

print("Carregando banco de questões...")
df = pd.read_json(local_file)

# Limpeza básica do dataset (removendo nulos e indefinidos)
df.dropna(inplace=True)
df = df[df['area'] != 'UNDEFINED']
df.reset_index(drop=True, inplace=True)
print(f"Banco de dados carregado com {len(df)} questões.")

# 2. Inicializa e treina o Vetorizador TF-IDF (mesma configuração do seu notebook)
print("Vetorizando o banco de dados (gerando matriz TF-IDF)...")
tfidf = TfidfVectorizer(
    min_df=2, 
    encoding='utf-8', 
    ngram_range=(1, 2),
    stop_words=stopwords.words('portuguese')
)

# Transforma a coluna de enunciados na nossa matriz de features
features = tfidf.fit_transform(df['enunciado'].values.astype('U'))

def buscar_questoes_similares(duvida_usuario, top_n=3):
    """
    Recebe uma frase/dúvida do usuário e retorna as top_n questões mais similares
    do banco do ENEM com base na Similaridade de Cosseno.
    """
    # Transforma a dúvida do usuário no mesmo espaço vetorial TF-IDF
    vetor_duvida = tfidf.transform([duvida_usuario])
    
    # Calcula o cosseno do ângulo entre a dúvida e todas as questões do banco
    similaridades = cosine_similarity(vetor_duvida, features).flatten()
    
    # Pega os índices das top_n questões com maior similaridade
    top_indices = similaridades.argsort()[-top_n:][::-1]
    
    resultados = []
    for idx in top_indices:
        resultados.append({
            "enunciado": df.iloc[idx]['enunciado'],
            "area": df.iloc[idx]['area'],
            "similaridade": similaridades[idx]
        })
        
    return resultados

# 3. Teste de Funcionamento
if __name__ == "__main__":
    print("\n" + "="*50)
    print(" SISTEMA DE BUSCA SEMÂNTICA — ENEM ")
    print("="*50)
    
    # Exemplo de teste
    teste_query = "Qual a relação entre o feudalismo e a suserania e vassalagem na Idade Média?"
    print(f"\nDúvida de Teste: '{teste_query}'")
    
    print("\nBuscando questões similares...")
    questoes = buscar_questoes_similares(teste_query, top_n=2)
    
    for i, q in enumerate(questoes, 1):
        print(f"\n[{i}] Similaridade: {q['similaridade']:.4f} | Área: {q['area'].upper()}")
        print("-" * 40)
        # Limita a exibição do enunciado para não poluir o terminal
        enunciado_resumido = q['enunciado'][:400] + "..." if len(q['enunciado']) > 400 else q['enunciado']
        print(enunciado_resumido)
        print("-" * 40)
