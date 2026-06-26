import subprocess
import sys

# Garante que as dependências necessárias estão instaladas
for package in ['bs4', 'requests']:
    try:
        if package == 'bs4':
            import bs4
        else:
            import requests
    except ImportError:
        print(f"Instalando {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

import requests
from bs4 import BeautifulSoup
import urllib.parse

def buscar_materias_brasil_escola(termo_busca):
    """
    Realiza uma busca por temas no Brasil Escola utilizando o Yahoo Search
    como motor de busca (altamente estável, sem JavaScript obrigatório e sem CAPTCHAs).
    Retorna os títulos e links dos 3 artigos mais relevantes.
    """
    termo_formatado = urllib.parse.quote_plus(f"site:brasilescola.uol.com.br {termo_busca}")
    url = f"https://search.yahoo.com/search?p={termo_formatado}"
    
    # Cabeçalho padrão para simular navegador real
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    
    print(f"Buscando via Yahoo: {url}")
    try:
        resposta = requests.get(url, headers=headers, timeout=12)
    except Exception as e:
        print(f"Erro ao conectar ao motor de busca: {e}")
        return []
    
    if resposta.status_code != 200:
        print(f"Erro na requisição: Status {resposta.status_code}")
        return []
        
    soup = BeautifulSoup(resposta.text, 'html.parser')
    resultados = []
    
    # No Yahoo Search, os resultados orgânicos vêm em blocos de título <h3>
    for h3_tag in soup.find_all('h3'):
        link_tag = h3_tag.find('a')
        if link_tag:
            href = link_tag.get('href', '')
            titulo = link_tag.text.strip()
            
            # Limpa espaços em branco duplicados
            titulo = " ".join(titulo.split())
            
            # Filtra apenas links de artigos válidos do Brasil Escola
            if "brasilescola.uol.com.br" in href and "yahoo.com" not in href:
                # Evita duplicados e títulos vazios
                if len(titulo) > 5 and not any(r['link'] == href for r in resultados):
                    resultados.append({
                        "titulo": titulo,
                        "link": href
                    })
                
    return resultados[:3] # Retorna as 3 matérias mais relevantes

if __name__ == "__main__":
    print("="*50)
    print(" TESTE DE WEB SCRAPING — BRASIL ESCOLA ")
    print("="*50)
    
    termo = "feudalismo suserania vassalagem"
    print(f"\nTermo de busca: '{termo}'")
    
    materias = buscar_materias_brasil_escola(termo)
    
    print(f"\nResultados encontrados ({len(materias)}):")
    print("-" * 50)
    for i, m in enumerate(materias, 1):
        print(f"[{i}] {m['titulo']}")
        print(f"    Link: {m['link']}")
        print("-" * 50)
