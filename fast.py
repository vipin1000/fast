import os
import time
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from bs4 import BeautifulSoup
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings




# === Configuration ===
STATIC_URLS = [
    # "https://www.velocis.in/",
    # "https://www.velocis.in/about",
    # "https://www.velocis.in/meet-the-leaders",
    # "https://www.velocis.in/domains/enterprise-network",
    # "https://www.velocis.in/domains/cybersecurity",
    # "https://www.velocis.in/domains/digital-workplace",
    # "https://www.velocis.in/domains/data-centre",
    # "https://www.velocis.in/domains/public-cloud",
    # "https://www.velocis.in/domains/application-transformation",
    # "https://www.velocis.in/products/bullseye",
    # "https://www.velocis.in/products/observer",
    # "https://www.velocis.in/products/compliance-engine",
    # "https://www.velocis.in/services/customer-experience-services",
    # "https://www.velocis.in/services/engineering-services",
    # "https://www.velocis.in/services/managed-services",
    # "https://www.velocis.in/services/professional-services",
    # "https://www.velocis.in/careers",
    # "https://www.velocis.in/life-at-velocis",
    # "https://www.velocis.in/blog",
    # "https://www.velocis.in/case-studies",
    # "https://www.velocis.in/contact-us",
    # "https://www.velocis.in/listing",
    "https://asdf-self-delta.vercel.app/",
    "https://asdf-self-delta.vercel.app/contact.html",
    "https://asdf-self-delta.vercel.app/properties.html",
    "https://asdf-self-delta.vercel.app/services.html",
    "https://asdf-self-delta.vercel.app/about.html"
]  
API_URL = "https://7649cc6059e9.ngrok-free.app/generate"  # Update as needed
CHROMA_INDEX_PATH = "chroma_index"
REBUILD_EMBEDDINGS = True # Set True to rebuild embeddings, False to load existing

app = FastAPI(title="RAG Backend API")

# === CORS ===
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Data Models ===
class QueryRequest(BaseModel):
    query: str
    

class QueryResponse(BaseModel):
    response: str
   

# === Global Vectorstore ===
VECTORSTORE = None

# === Scraping ===
def extract_visible_text_with_requests(url: str) -> str:
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, "html.parser")
        
        # Remove script and style elements
        for tag in soup(["script", "style", "noscript", "nav", "footer"]):
            tag.decompose()
        
        # Extract text from main content areas
        main_content = ""
        
        # Try to find main content areas
        for selector in ['main', 'article', '.content', '#content', '.main', '#main']:
            elements = soup.select(selector)
            if elements:
                main_content += " ".join([elem.get_text(separator="\n") for elem in elements])
        
        # If no main content found, get all text
        if not main_content:
            main_content = soup.get_text(separator="\n")
        
        # Clean up the text
        lines = [line.strip() for line in main_content.splitlines() if line.strip()]
        return "\n".join(lines)
        
    except Exception as e:
        print(f"Error scraping {url}: {e}")
        return f"Error accessing {url}: {str(e)}"

# === Chroma Load or Build ===
def load_or_build_vectorstore():
    # Use a smaller model to reduce memory usage
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    if os.path.exists(CHROMA_INDEX_PATH) and not REBUILD_EMBEDDINGS:
        print("🔍 Loading Chroma index from disk...")
        return Chroma(persist_directory=CHROMA_INDEX_PATH, embedding_function=embeddings)
    else:
        print("🚀 Building Chroma index from scratch...")
        documents = []
        for url in STATIC_URLS:
            text = extract_visible_text_with_requests(url)
            if text:
                documents.append(Document(page_content=text, metadata={"source": url}))
        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
        chunks = splitter.split_documents(documents)
        vectorstore = Chroma.from_documents(chunks, embedding=embeddings, persist_directory=CHROMA_INDEX_PATH)
        vectorstore.persist()
        return vectorstore

# === Retrieval ===
def retrieve_relevant_context(query, vectorstore, top_k=5):
    retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})
    results = retriever.invoke(query)
    return "\n\n".join([doc.page_content for doc in results])

# === Initialize on Startup ===
@app.on_event("startup")
async def startup_event():
    global VECTORSTORE
    print("🚀 Starting up RAG Backend API...")
    print("📚 Vectorstore will be loaded on first request to save memory")
    # Don't load anything heavy during startup

# === Health Check ===
@app.get("/")
def health_check():
    return {"status": "healthy", "message": "RAG Backend API is running", "vectorstore_loaded": VECTORSTORE is not None}

@app.get("/test")
def test_endpoint():
    return {"message": "API is working!", "timestamp": time.time()}

# === Query Endpoint ===
@app.post("/chat/query", response_model=QueryResponse)
def process_query(payload: QueryRequest):
    global VECTORSTORE
    try:
        if VECTORSTORE is None:
            print("🔄 Loading vectorstore on first request...")
            VECTORSTORE = load_or_build_vectorstore()
            print("✅ Vectorstore loaded successfully!")

        context = retrieve_relevant_context(payload.query, VECTORSTORE)
    except Exception as e:
        print(f"❌ Error loading vectorstore: {e}")
        raise HTTPException(status_code=500, detail=f"Vectorstore error: {str(e)}")

    full_context = (
        "\n\n[Relevant Context:]\n" + context +
        "\n\n[Scraped URLs:]\n" + "\n".join(STATIC_URLS)
    )

    api_payload = {"data": full_context, "query": payload.query}

    try:
        response = requests.post(API_URL, json=api_payload)
        response.raise_for_status()
        ai_answer = response.json()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference API error: {e}")

    return QueryResponse(response=ai_answer)

# Run with: uvicorn app:app --reload
