# Usar uma imagem base oficial do Python leve
FROM python:3.9-slim

# Definir o diretório de trabalho dentro do container
WORKDIR /code

# Copiar o arquivo de dependências primeiro para aproveitar o cache do Docker
COPY ./requirements.txt /code/requirements.txt

# Instalar as dependências do Python
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Baixar as stopwords do NLTK durante o build para evitar latência no startup
RUN python -m nltk.downloader stopwords

# Copiar todos os arquivos do projeto para o container
COPY . /code

# Configurar permissões de leitura/escrita necessárias para o Hugging Face Spaces
RUN chmod -R 777 /code

# Porta padrão exposta pelo Hugging Face Spaces é a 7860
EXPOSE 7860

# Comando para iniciar o servidor uvicorn na porta do Hugging Face Spaces (7860)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
