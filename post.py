import requests
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings





# === Configuration ===
API_URL = "https://cac697cb282d.ngrok-free.app/generate"
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
def build_vectorstore_from_static_urls():
    all_docs = []

    for url in STATIC_URLS:
        try:
            loader = WebBaseLoader(
                url
            )
            all_docs.extend(loader.load())
        except Exception as e:
            print(f"⚠️ Failed to load {url}: {e}")

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(all_docs)

    embeddings = HuggingFaceEmbeddings()
    vectorstore = FAISS.from_documents(chunks, embedding=embeddings)
    return vectorstore



def format_chat_history(chat_history, max_turns=3):
    recent_history = chat_history[-max_turns:]
    return "\n".join([f"User: {h['user']}\nAI: {h['ai']}" for h in recent_history])


# === Retrieve Context ===
def retrieve_relevant_context(query, vectorstore, top_k=3):
    retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})
    results = retriever.invoke(query)
    return "\n\n".join([doc.page_content for doc in results])

# === Main Execution ===
def main():
    print("[🚀] Building vector store from predefined URLs...")
    vs = build_vectorstore_from_static_urls()

    chat_history = []

    while True:
        query = input("\n🔍 Enter your query (or type 'exit' to quit): ").strip()
        if query.lower() == "exit":
            break

        print("[🔎] Retrieving relevant context...")
        context_chunk = retrieve_relevant_context(query, vs)

        # Format history and append to context
        formatted_history = format_chat_history(chat_history)
        full_context = (
            "\n\n[Scraped URLs:]\n" + "\n".join(STATIC_URLS)+
            "\n\n[Conversation History:]\n" + formatted_history + 
            "\n\n[Relevant Context:]\n" + context_chunk  
            
        )

        payload = {
            "data": full_context,
            "query": query
        }

        try:
            print("[📡] Sending payload to FastAPI inference endpoint...")
            response = requests.post(API_URL, json=payload)
            response.raise_for_status()
            answer = response.json()

            print("\n✅ AI Response:\n", answer)

            # Update chat history buffer
            chat_history.append({"user": query, "ai": answer})

        except requests.exceptions.RequestException as e:
            print("❌ Request failed:", e)



if __name__=="__main__":
    main()       