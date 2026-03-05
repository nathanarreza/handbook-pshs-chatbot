
!apt-get install -y zstd

!apt-get update && apt-get install -y pciutils zstd
!curl -fsSL https://ollama.com/install.sh | sh
!pip install llama-index-llms-ollama llama-index-embeddings-ollama llama-index fastapi uvicorn nest-asyncio pyngrok python-multipart

!apt-get install -y zstd

!curl -fsSL https://ollama.com/install.sh | sh

import subprocess
import time
import os

!pkill ollama

# Start the server and send logs to a file
with open("ollama_logs.txt", "w") as f:
    subprocess.Popen(["ollama", "serve"], stdout=f, stderr=f)

time.sleep(10)

print("Downloading Llama 3.2 (1B)...")
!ollama pull llama3.2:1b
print("Downloading Embedding Model...")
!ollama pull nomic-embed-text

os.environ['OLLAMA_HOST'] = '0.0.0.0:11434'
subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

time.sleep(5)

!ollama pull llama3.2:1b
!ollama pull all-minilm

import os
import nest_asyncio
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, StorageContext, load_index_from_storage, PromptTemplate
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core.node_parser import SentenceSplitter
import uvicorn
import threading

nest_asyncio.apply()
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


Settings.chunk_size = 256
Settings.chunk_overlap = 50

Settings.llm = Ollama(
    model="llama3.2:1b",
    request_timeout=60.0,
    base_url="http://localhost:11434",
    temperature=0.1,
    additional_kwargs={
        "num_predict": 300
    }
)

Settings.embed_model = OllamaEmbedding(
    model_name="nomic-embed-text",
    base_url="http://localhost:11434",
    embed_batch_size=10
)

query_engine = None
PERSIST_DIR = "./storage"

def initialize_index():
    global query_engine

    pdf_files =["pshs-handbook.pdf", "coc-handbook.pdf"]

    for pdf in pdf_files:
        if not os.path.exists(pdf):
            print(f"Missing file: {pdf}. Please upload it to your workspace.")
            return

    try:
        if not os.path.exists(PERSIST_DIR):
            print(f"Creating new index from {len(pdf_files)} files... (This may take a moment)")

            documents = SimpleDirectoryReader(input_files=pdf_files).load_data()


            splitter = SentenceSplitter(chunk_size=256, chunk_overlap=50)

            index = VectorStoreIndex.from_documents(
                documents,
                transformations=[splitter]
            )
            index.storage_context.persist(persist_dir=PERSIST_DIR)
        else:
            print("Loading existing index from storage... (Lightning fast!)")
            storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
            index = load_index_from_storage(storage_context)

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
        print("Handbook and additional documents ready!")

    except Exception as e:
        print(f"Error index: {e}")

initialize_index()

@app.get("/ask")
async def ask(question: str = Query(...)):
    global query_engine
    if query_engine is None:
        return {"answer": "Server not ready."}
    try:
        print(f"Question received: {question}")
        response = await query_engine.aquery(question)

        print(f"Answered successfully.")
        return {"answer": str(response)}
    except Exception as e:
        print(f"Server error: {str(e)}")
        return {"answer": f"Backend Error: {str(e)}"}

def run():
    uvicorn.run(app, host="0.0.0.0", port=8000)


threading.Thread(target=run, daemon=True).start()

!npm install -g localtunnel

!npx lt --port 8000 --subdomain pshscbzrc-school-handbook-chat

!fuser -k 8000/tcp