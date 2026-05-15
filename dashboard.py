import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader, PyPDFLoader
import glob
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()

# ── Page Config ──────────────────────────────
st.set_page_config(
    page_title="Crisis Analyzer Bot",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Crisis Analyzer Bot")
st.markdown("*AI-powered analysis of historical financial crises*")
st.divider()

# ── Load bot only once (cached for speed) ────
@st.cache_resource
def load_bot():
    documents = []
    data_folder = "data"
    
    for txt_file in glob.glob(f"{data_folder}/*.txt"):
        loader = TextLoader(txt_file, encoding="utf-8")
        documents.extend(loader.load())
    
    for pdf_file in glob.glob(f"{data_folder}/*.pdf"):
        loader = PyPDFLoader(pdf_file)
        documents.extend(loader.load())
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)
    
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vectorstore = FAISS.from_documents(chunks, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 6})
    
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0.3)
    
    prompt = ChatPromptTemplate.from_template(
        "You are a financial crisis expert analyzing historical economic crises.\n"
        "Use the context below to provide a detailed, well-structured answer.\n"
        "If comparing crises, use bullet points or clear sections.\n"
        "If the context has partial information, share what's available "
        "and note what's missing.\n\n"
        "Previous conversation:\n{history}\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n\n"
        "Detailed Answer:"
    )
    
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    def format_history(history):
        if not history:
            return "No previous conversation."
        formatted = []
        for msg in history[-6:]:
            role = "User" if isinstance(msg, HumanMessage) else "Bot"
            formatted.append(f"{role}: {msg.content}")
        return "\n".join(formatted)
    
    rag_chain = (
        RunnablePassthrough.assign(
            context=lambda x: format_docs(retriever.invoke(x["question"])),
            history=lambda x: format_history(x.get("history", []))
        )
        | prompt
        | llm
        | StrOutputParser()
    )
    
    chain = RunnableParallel(
        answer=rag_chain,
        sources=lambda x: retriever.invoke(x["question"])
    )
    
    return chain, len(documents), len(chunks)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
if "history" not in st.session_state:
    st.session_state.history = []

# ── Sidebar ─────────────────────────────────
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
    This bot analyzes historical financial crises using RAG (Retrieval Augmented Generation).
    
    **Tech Stack:**
    - 🐍 Python
    - 🔗 LangChain
    - 🧠 Google Gemini
    - 📊 FAISS Vector DB
    """)
    
    st.divider()
    
    st.header("💡 Try Asking:")
    st.markdown("""
    - What caused the 2008 crisis?
    - Compare 2008 and COVID crises
    - Which crisis recovered faster?
    - Early warning signs of crises
    - Sectors affected by COVID
    """)
    
    st.divider()
    
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.session_state.history = []
        st.rerun()

# ── Load Bot ────────────────────────────────
with st.spinner("🚀 Loading crisis data and AI model..."):
    chain, num_docs, num_chunks = load_bot()

st.success(f"✅ Loaded {num_docs} documents → {num_chunks} chunks")

# ── Display Chat History ────────────────────
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message:
            with st.expander("📚 Sources"):
                for src in message["sources"]:
                    st.markdown(f"- `{src}`")

# ── Chat Input ──────────────────────────────
if question := st.chat_input("Ask anything about financial crises..."):
    # Show user message
    with st.chat_message("user"):
        st.markdown(question)
    st.session_state.messages.append({"role": "user", "content": question})
    
    # Get bot response
    with st.chat_message("assistant"):
        with st.spinner("🔍 Thinking..."):
            result = chain.invoke({
                "question": question,
                "history": st.session_state.history
            })
            
            answer = result['answer']
            
            # Get unique sources
            unique_sources = set()
            for doc in result['sources']:
                source = doc.metadata.get('source', 'Unknown')
                unique_sources.add(source)
            
            sources = list(unique_sources)
            
            # Display answer
            st.markdown(answer)
            
            # Display sources
            with st.expander("📚 Sources"):
                for src in sources:
                    st.markdown(f"- `{src}`")
    
    # Save to memory
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources
    })
    st.session_state.history.append(HumanMessage(content=question))
    st.session_state.history.append(AIMessage(content=answer))