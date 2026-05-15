import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader, PyPDFLoader
import os
import glob
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

# ── Step 1: Load ALL documents (TXT + PDF) ────────
print("📄 Loading crisis data...")

documents = []
data_folder = "data"  # we'll put all docs in a "data" folder

# Load all TXT files
for txt_file in glob.glob(f"{data_folder}/*.txt"):
    print(f"   → Loading {txt_file}")
    loader = TextLoader(txt_file, encoding="utf-8")
    documents.extend(loader.load())

# Load all PDF files
for pdf_file in glob.glob(f"{data_folder}/*.pdf"):
    print(f"   → Loading {pdf_file}")
    loader = PyPDFLoader(pdf_file)
    documents.extend(loader.load())

if len(documents) == 0:
    print("❌ No documents found in 'data' folder!")
    print("   Add .txt or .pdf files to the data/ folder and try again.")
    exit()

print(f"✅ Loaded {len(documents)} document chunk(s) from data folder\n")

# ── Step 2: Split into chunks ─────────────────────
print("✂️  Splitting into chunks...")
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(documents)
print(f"✅ Created {len(chunks)} chunks\n")

# ── Step 3: Convert chunks to vectors ─────────────
print("🧠 Creating embeddings (first run takes ~30 sec)...")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(chunks, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 6})
print("✅ Vector database ready\n")

# ── Step 4: Set up the LLM ────────────────────────
print("🚀 Connecting to Gemini...")
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0.3)

# ── Step 5: Build the prompt and chain ────────────
prompt = ChatPromptTemplate.from_template(
    "You are a financial crisis expert analyzing historical economic crises.\n"
    "Use the context below to provide a detailed, well-structured answer.\n"
    "If comparing crises, use bullet points or clear sections.\n"
    "If the context has partial information, share what's available "
    "and note what's missing — don't refuse to answer entirely.\n\n"
    "Context:\n{context}\n\n"
    "Question: {question}\n\n"
    "Detailed Answer:"
)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Build chain that returns BOTH answer and source documents
from langchain_core.runnables import RunnableParallel

rag_chain = (
    RunnablePassthrough.assign(
        context=lambda x: format_docs(retriever.invoke(x["question"]))
    )
    | prompt
    | llm
    | StrOutputParser()
)

chain = RunnableParallel(
    answer=rag_chain,
    sources=lambda x: retriever.invoke(x["question"])
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
    result = chain.invoke({"question": question})
    
    # Print the answer
    print(f"\n💡 Answer: {result['answer']}")
    
    # Print unique sources used
    print("\n📚 Sources used:")
    unique_sources = set()
    for doc in result['sources']:
        source = doc.metadata.get('source', 'Unknown')
        unique_sources.add(source)
    
    for i, source in enumerate(unique_sources, 1):
        print(f"   {i}. {source}")