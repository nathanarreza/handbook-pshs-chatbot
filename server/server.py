import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, StorageContext, load_index_from_storage, PromptTemplate
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core.node_parser import SentenceSplitter

app = Flask(__name__)
CORS(app)  # Allows your frontend to connect to this backend

# --- CONFIGURATION ---
PERSIST_DIR = "./storage"
PDF_FILES = ["pshs-handbook.pdf", "coc-handbook.pdf"]

# Initialize Models
Settings.llm = Ollama(
    model="llama3.2:1b",
    request_timeout=120.0,
    base_url="http://localhost:11434",
    temperature=0.1,
)

Settings.embed_model = OllamaEmbedding(
    model_name="nomic-embed-text",
    base_url="http://localhost:11434",
)

Settings.chunk_size = 256
Settings.chunk_overlap = 50

# Global variable for the query engine
query_engine = None

def initialize_index():
    global query_engine

    # Check if PDFs exist
    for pdf in PDF_FILES:
        if not os.path.exists(pdf):
            print(f"CRITICAL: {pdf} not found in the current directory.")
            return

    try:
        if not os.path.exists(PERSIST_DIR):
            print("Creating new index from documents...")
            documents = SimpleDirectoryReader(input_files=PDF_FILES).load_data()
            splitter = SentenceSplitter(chunk_size=256, chunk_overlap=50)
            index = VectorStoreIndex.from_documents(documents, transformations=[splitter])
            index.storage_context.persist(persist_dir=PERSIST_DIR)
        else:
            print("Loading existing index from storage...")
            storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
            index = load_index_from_storage(storage_context)

        # Custom Prompt
        qa_prompt_tmpl_str = (
            "You are a helpful and friendly school assistant. You are answering questions based on excerpts from the official school documents below.\n"
            "---------------------\n"
            "{context_str}\n"
            "---------------------\n"
            "Given the document information and not prior knowledge, answer the query.\n"
            "IMPORTANT RULES:\n"
            "1. NEVER use the phrases 'based on the context' or 'context provided'.\n"
            "2. If you refer to the source, always say 'Based on the school documents' or 'According to the handbook'.\n"
            "Query: {query_str}\n"
            "Answer: "
        )
        qa_prompt_tmpl = PromptTemplate(qa_prompt_tmpl_str)

        query_engine = index.as_query_engine(
            similarity_top_k=4,
            text_qa_template=qa_prompt_tmpl
        )
        print("Index ready!")

    except Exception as e:
        print(f"Error during index initialization: {e}")

# Run initialization before the first request
initialize_index()

@app.route('/ask', methods=['GET'])
def ask():
    question = request.args.get('question')
    if not question:
        return jsonify({"error": "No question provided"}), 400

    if query_engine is None:
        return jsonify({"answer": "Server is still initializing. Please try again in a moment."}), 503

    try:
        print(f"Processing question: {question}")
        # Note: We use .query() (sync) instead of .aquery() (async) for standard Flask
        response = query_engine.query(question)
        return jsonify({"answer": str(response)})
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"answer": f"Internal Server Error: {str(e)}"}), 500

if __name__ == '__main__':
    # Run the Flask app on localhost:8000
    app.run(host='0.0.0.0', port=8000, debug=False)
