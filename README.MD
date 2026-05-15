# 📊 Crisis Analyzer Bot

> AI-powered chatbot that analyzes historical financial crises using RAG (Retrieval Augmented Generation) and Google Gemini.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![LangChain](https://img.shields.io/badge/LangChain-RAG-green)
![Gemini](https://img.shields.io/badge/Google-Gemini-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-Live-red)

---

## 🌐 Live Demo

**Try it now →** [https://rv-crisis-analyzer.streamlit.app/](https://rv-crisis-analyzer.streamlit.app/)

*No setup required — runs directly in your browser!*

---

## 🎯 What This Does

An AI chatbot that reads historical financial crisis documents (2008, COVID-19, etc.) and answers questions about them with verifiable sources. Built with RAG architecture — meaning every answer is grounded in actual documents, not hallucinated.

## ✨ Features

- 🤖 **AI-powered analysis** — Powered by Google Gemini 2.5
- 📚 **Multi-document support** — Reads both TXT and PDF files
- 🔍 **Smart retrieval** — FAISS vector database for semantic search
- 💬 **Conversation memory** — Remembers context across questions
- 📍 **Source tracking** — Every answer cites its sources
- 🌐 **Web interface** — Clean ChatGPT-like UI built with Streamlit
- ☁️ **Cloud deployed** — Live on Streamlit Cloud

## 🛠️ Tech Stack

- **Python 3.11**
- **LangChain** — RAG framework
- **Google Gemini API** — LLM
- **FAISS** — Vector database for fast similarity search
- **HuggingFace Embeddings** — Sentence transformers for text vectorization
- **Streamlit** — Web UI framework
- **PyPDF** — PDF document loader

## 🧠 How RAG Works