import os
import time
import requests
import streamlit as st
from bs4 import BeautifulSoup
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

# ========== Configurations ==========
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
    "https://www.velocis.in/events",
]

API_URL = "https://646869779f1b.ngrok-free.app/generate"  # Replace accordingly


# ========== Web Scraper using Selenium ==========
@st.cache_data(show_spinner="🔄 Scraping dynamic content...")
def extract_visible_text_with_selenium(url: str) -> str:
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    driver = webdriver.Chrome(options=options)

    try:
        driver.get(url)
        time.sleep(3)  # Wait for JS to render
        html = driver.page_source
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        visible_text = soup.get_text(separator="\n")
        return "\n".join([line.strip() for line in visible_text.splitlines() if line.strip()])
    except Exception as e:
        return f"Error scraping {url}: {e}"
    finally:
        driver.quit()


# ========== Vector Store Builder ==========
@st.cache_resource(show_spinner="🔄 Embedding website content...")
def build_vectorstore():
    documents = []
    for url in STATIC_URLS:
        text = extract_visible_text_with_selenium(url)
        if text:
            documents.append(Document(page_content=text, metadata={"source": url}))
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_documents(documents)
    embeddings = HuggingFaceEmbeddings()
    return FAISS.from_documents(chunks, embedding=embeddings)


# ========== Retriever ==========
def retrieve_relevant_context(query, vectorstore, top_k=5):
    retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})
    results = retriever.invoke(query)
    return "\n\n".join([doc.page_content for doc in results])


# ========== Chat History Formatter ==========
def format_chat_history(chat_history, max_turns=4):
    recent = chat_history[-max_turns:]
    return "\n".join([f"User: {h['user']}\nAI: {h['ai']}" for h in recent])


# ========== Streamlit App ==========
st.set_page_config(page_title="RAG ", layout="wide")
st.title("🧠 Dynamic RAG")
st.markdown("Chat with Website.")

# Initialize chat history
if "history" not in st.session_state:
    st.session_state.history = []

# Build vectorstore once
vectorstore = build_vectorstore()

# User input
query = st.text_input("💬 Enter your query", placeholder="e.g., Contact Info of Velocis")
submit = st.button("Ask")

if submit and query:
    with st.spinner("🔍 Processing your query..."):
        # Retrieve context
        context = retrieve_relevant_context(query, vectorstore)
        history_text = format_chat_history(st.session_state.history)
        full_context = (
            
            "\n\n[Relevant Context:]\n" + context +
            "\n\n[Scraped URLs:]\n" + "\n".join(STATIC_URLS)
        )

        # Compose payload
        payload = {"data": full_context, "query": query}

        # Hit inference endpoint
        try:
            response = requests.post(API_URL, json=payload)
            response.raise_for_status()
            ai_answer = response.json()
            st.session_state.history.append({"user": query, "ai": ai_answer})
            st.success("✅ Response received.")
            st.markdown(f"**🤖 AI:** {ai_answer}")
        except Exception as e:
            st.error(f"❌ Error: {e}")

# Chat history display
if st.session_state.history:
    with st.expander("🗂️ Chat History"):
        for turn in st.session_state.history:
            st.markdown(f"**You:** {turn['user']}")
            st.markdown(f"**AI:** {turn['ai']}")
