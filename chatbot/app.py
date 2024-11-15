import openai
import numpy as np
from flask import Flask, render_template, request, jsonify, session
from sklearn.metrics.pairwise import cosine_similarity
import os
from dotenv import load_dotenv

app = Flask(__name__)
app.secret_key = "your_secret_key"

# Carregar variáveis de ambiente a partir do arquivo .env
load_dotenv()

# Obter a chave da API diretamente do arquivo .env
api_key = os.getenv("OPENAI_API_KEY")

# Temporary cache to store embeddings
corpus_embeddings_cache = {}
question_embeddings_cache = {}

# Load and preprocess the document content from 'output.txt'
def load_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

# Load the document content
corpus_text = load_text_file('output.txt')

# Function to obtain embeddings with caching
def get_embedding(text, api_key, model="text-embedding-3-large"):
    openai.api_key = api_key  # Set the API key globally
    try:
        if text in corpus_embeddings_cache:
            return corpus_embeddings_cache[text]
        
        # Get the embedding from OpenAI API without api_key as an argument in create()
        response = openai.embeddings.create(input=[text], model=model)
        embedding = response.data[0].embedding  # Access the embedding directly as an attribute
        
        # Cache the embedding
        corpus_embeddings_cache[text] = embedding
        return embedding
    except Exception as e:
        print(f"Error in get_embedding: {str(e)}")  # Logging the error
        return f"An error occurred: {str(e)}"

# Split text into chunks and generate embeddings for each chunk
def preprocess_corpus(api_key, chunk_size=1000):
    chunked_corpus = [corpus_text[i:i + chunk_size] for i in range(0, len(corpus_text), chunk_size)]
    embeddings = []
    for chunk in chunked_corpus:
        embedding = get_embedding(chunk, api_key)
        if isinstance(embedding, str):  # If an error string is returned, stop processing
            print(f"Error during corpus preprocessing: {embedding}")  # Log the error
            return embedding
        embeddings.append(embedding)
    return chunked_corpus, np.array(embeddings)

# Cache corpus embeddings on first request
chunked_corpus = None
corpus_embeddings = None

# Retrieve question embedding with caching
def get_question_embedding(question, api_key):
    if question in question_embeddings_cache:
        return question_embeddings_cache[question]
    
    question_embedding = get_embedding(question, api_key)
    question_embeddings_cache[question] = question_embedding
    return question_embedding

def find_similar_response(question, api_key, max_chunks=30):
    global chunked_corpus, corpus_embeddings

    # Initialize corpus embeddings if not done already
    if chunked_corpus is None or corpus_embeddings is None:
        preprocessed_corpus = preprocess_corpus(api_key)
        if isinstance(preprocessed_corpus, str):  # Check if an error occurred
            return preprocessed_corpus
        chunked_corpus, corpus_embeddings = preprocessed_corpus

    question_embedding = get_question_embedding(question, api_key)
    if isinstance(question_embedding, str):  # Check for error in question embedding
        return question_embedding
    
    similarities = cosine_similarity([question_embedding], corpus_embeddings)[0]
    best_indexes = np.argsort(similarities)[::-1][:max_chunks]
    
    # Gather the most relevant chunks as context
    context = "\n".join(chunked_corpus[idx] for idx in best_indexes)
    
    # Chamada à API de completions atualizada
    response = openai.chat.completions.create(
        model="gpt-4-turbo",
        messages=[ 
            {
                "role": "system",
                "content": (
                    "Only english."
                    "You are a knowledgeable assistant with access to a detailed document database on topics related to market intelligence. "
                    "Use only the provided context to answer the user's questions as accurately and specifically as possible. "
                    "If you can't generate a answear based on the context, respond with 'I cannot determine this based on the available data.' "
                    "If the user asks about trends in market intelligence or marketing, analyze only articles from 2024 and respond with: 'Based on the database, the trends are...' followed by a bulleted list of each trend with a brief description and some article name."
                )
            },
            {"role": "user", "content": f"Context: {context}"},
            {"role": "user", "content": question}
        ],
        temperature=1,
        max_tokens=4096,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )

    # Acessar diretamente usando atributos em vez de subscrição
    return response.choices[0].message.content

@app.route('/')
def home():
    if "conversation_history" not in session:
        session["conversation_history"] = []
    return render_template('chat.html')

@app.route('/ask', methods=['POST'])
def ask():
    data = request.json
    user_question = data.get("question")
    
    # Use the api_key from the .env file directly without needing to send it in the request
    if not api_key:
        return jsonify({"error": "API key not found. Please configure the key in the .env file."}), 400

    # Limit conversation history to recent messages to avoid session overflow
    conversation_history = session.get("conversation_history", [])[-5:]
    conversation_history.append({"role": "user", "content": user_question})
    
    response_text = find_similar_response(user_question, api_key)
    conversation_history.append({"role": "assistant", "content": response_text})
    
    # Update the session with limited conversation history
    session["conversation_history"] = conversation_history
    
    return jsonify({"answer": response_text})

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
