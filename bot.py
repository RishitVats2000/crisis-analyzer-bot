import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Load API key from .env
load_dotenv()

print("🤖 Crisis Analyzer Bot — Starting up...\n")

# ── Step 1: Load the document ─────────────────────
print("📄 Loading crisis data...")
loader = TextLoader("crisis-2008.txt", encoding="utf-8")
documents = loader.load()
print(f"✅ Loaded {len(documents)} document(s)\n")

# ── Step 2: Split into chunks ─────────────────────
print("✂️  Splitting into chunks...")
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(documents)
print(f"✅ Created {len(chunks)} chunks\n")

# ── Step 3: Convert chunks to vectors ─────────────
print("🧠 Creating embeddings (first run takes ~30 sec)...")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(chunks, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
print("✅ Vector database ready\n")

# ── Step 4: Set up the LLM ────────────────────────
print("🚀 Connecting to Gemini...")
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0.3)

# ── Step 5: Build the prompt and chain ────────────
prompt = ChatPromptTemplate.from_template(
    "You are a financial crisis expert. Answer the question based ONLY on the context below. "
    "If the answer isn't in the context, say 'I don't have that information.'\n\n"
    "Context:\n{context}\n\n"
    "Question: {question}\n\n"
    "Answer:"
)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

print("✅ Bot ready!\n")

print("="*60)
print("Ask anything about the 2008 financial crisis!")
print("Type 'quit' to exit")
print("="*60)

while True:
    question = input("\n❓ Your question: ").strip()
    if question.lower() in ['quit', 'exit', 'q']:
        print("\n👋 Goodbye!")
        break
    if not question:
        continue
    
    print("\n🔍 Thinking...")
    answer = chain.invoke(question)
    print(f"\n💡 Answer: {answer}")