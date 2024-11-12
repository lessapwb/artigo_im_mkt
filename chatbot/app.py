import openai
import numpy as np
from flask import Flask, render_template, request, jsonify
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
import os

app = Flask(__name__)

# Carregar a chave da API do arquivo .env
load_dotenv()

# Função para carregar o conteúdo do arquivo .txt
def load_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

# Carregar os textos do arquivo .txt
corpus_text = load_text_file('output.txt')  # Caminho relativo

# Função para obter o embedding de um texto usando o modelo text-embedding-3-large
def get_embedding(text, api_key):
    openai.api_key = api_key  # Usar a chave da API fornecida
    try:
        # Chama a API para gerar o embedding do texto
        response = openai.embeddings.create(
            model="text-embedding-3-large",  # Usando o modelo text-embedding-3-large
            input=text
        )
        # Acessa o embedding da resposta corretamente
        embedding = response.data[0].embedding  # Correção para acessar o 'embedding'
        return embedding
    except Exception as e:
        # Em caso de erro, retorna uma mensagem de erro
        return f"An error occurred: {str(e)}"

# Função para dividir o texto em pedaços menores para maximizar o uso de tokens
def chunk_text(text, chunk_size=3000):
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

# Função para encontrar respostas similares no corpus e gerar um contexto maior
def find_similar_responses(user_query, corpus_text, api_key, max_chunks=3):
    try:
        # Obter o embedding da pergunta do usuário
        query_embedding = get_embedding(user_query, api_key)

        if isinstance(query_embedding, str):  # Se a resposta for um erro, retornamos a mensagem
            return query_embedding

        # Carregar e dividir o texto do corpus em pedaços menores
        chunked_corpus = chunk_text(corpus_text)

        # Calcular a similaridade entre o embedding da pergunta e os pedaços do corpus
        corpus_embeddings = np.array([get_embedding(chunk, api_key) for chunk in chunked_corpus])

        # Verifica se algum erro foi gerado nos embeddings
        if isinstance(corpus_embeddings[0], str):  # Se algum embedding não for gerado corretamente
            return corpus_embeddings[0]  # Retorna o erro

        similarities = cosine_similarity([query_embedding], corpus_embeddings)[0]

        # Obter os índices dos textos mais similares
        best_indexes = np.argsort(similarities)[::-1][:max_chunks]

        # Construir o texto de contexto (pegando os 'max_chunks' mais relevantes)
        context = ""
        for index in best_indexes:
            context += chunked_corpus[index] + "\n"

        # Gerar uma resposta com o modelo de chat
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an expert analyst specializing in market intelligence. "
                        "You have access to a carefully curated database of articles. "
                        "Your responses should be based solely on the content of the provided articles. "
                        "If the question asks for information that cannot be derived from the articles, respond with 'I cannot determine this based on the available data.' "
                        "Ensure your answers are detailed, clear, and relevant."
                    )
                },
                {
                    "role": "user",
                    "content": f"Contextual background (for more precise analysis): {context}"
                },
                {
                    "role": "user",
                    "content": f"Specific question (to focus your answer): {user_query}"
                }
            ]
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        return f"<div>An error occurred: {str(e)}</div><button onclick='location.reload()'>Update the page</button>"

@app.route('/')
def home():
    return render_template('chat.html')

@app.route('/ask', methods=['POST'])
def ask():
    data = request.json
    user_question = data.get("question")
    api_key = request.headers.get("Authorization").split("Bearer ")[1]
    response_text = find_similar_responses(user_question, corpus_text, api_key)
    return jsonify({"answer": response_text})

if __name__ == "__main__":
    # Use a variável de ambiente PORT fornecida pelo Heroku
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))

