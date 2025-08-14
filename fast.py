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
from selenium import webdriver
from selenium.webdriver.chrome.options import Options



# === Configuration ===
STATIC_URLS = [
    "https://www.velocis.in/",
    "https://www.velocis.in/about",
    "https://www.velocis.in/meet-the-leaders",
    "https://www.velocis.in/domains/enterprise-network",
    "https://www.velocis.in/domains/cybersecurity",
    "https://www.velocis.in/domains/digital-workplace",
    "https://www.velocis.in/domains/data-centre",
    "https://www.velocis.in/domains/public-cloud",
    "https://www.velocis.in/domains/application-transformation",
    "https://www.velocis.in/products/bullseye",
    "https://www.velocis.in/products/observer",
    "https://www.velocis.in/products/compliance-engine",
    "https://www.velocis.in/services/customer-experience-services",
    "https://www.velocis.in/services/engineering-services",
    "https://www.velocis.in/services/managed-services",
    "https://www.velocis.in/services/professional-services",
    "https://www.velocis.in/careers",
    "https://www.velocis.in/life-at-velocis",
    "https://www.velocis.in/blog",
    "https://www.velocis.in/case-studies",
    "https://www.velocis.in/contact-us",
    "https://www.velocis.in/listing",
    "https://www.velocis.in/events"
]  
API_URL = "https://2ec875982955.ngrok-free.app/generate"  # Update as needed
CHROMA_INDEX_PATH = "chroma_index"
REBUILD_EMBEDDINGS = False # Set True to rebuild embeddings, False to load existing

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
def extract_visible_text_with_selenium(url: str) -> str:
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.add_argument("--remote-debugging-port=9222")
    options.add_argument("--disable-extensions")
    options.add_argument("--disable-plugins")
    options.add_argument("--disable-images")
    options.add_argument("--disable-javascript")
    
    # Use environment variables for Chrome paths on Render
    chrome_bin = os.getenv("CHROME_BIN", "/usr/bin/chromium")
    chromedriver_path = os.getenv("CHROMEDRIVER_PATH", "/usr/bin/chromedriver")
    
    options.binary_location = chrome_bin
    driver = webdriver.Chrome(options=options)
    try:
        driver.get(url)
        time.sleep(3)
        html = driver.page_source
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        visible_text = soup.get_text(separator="\n")
        return "\n".join([line.strip() for line in visible_text.splitlines() if line.strip()])
    finally:
        driver.quit()

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
            text = extract_visible_text_with_selenium(url)
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
    # Lazy load to reduce memory usage during startup
    print("🚀 Starting up RAG Backend API...")
    print("📚 Vectorstore will be loaded on first request to save memory")

# === Health Check ===
@app.get("/")
def health_check():
    return {"status": "healthy", "message": "RAG Backend API is running"}

# === Query Endpoint ===
@app.post("/chat/query", response_model=QueryResponse)
def process_query(payload: QueryRequest):
    global VECTORSTORE
    if VECTORSTORE is None:
        print("🔄 Loading vectorstore on first request...")
        VECTORSTORE = load_or_build_vectorstore()
        print("✅ Vectorstore loaded successfully!")

    context = retrieve_relevant_context(payload.query, VECTORSTORE)

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
