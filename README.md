# Nietzsche RAG API

## 📌 Overview

**Nietzsche RAG API** is a production-ready **Retrieval-Augmented Generation (RAG)** system built on Friedrich Nietzsche's philosophical works. It combines semantic search with vector embeddings to provide intelligent, context-aware answers from Nietzsche's writings.

The system indexes Nietzsche's major works including:
- **Thus Spoke Zarathustra**
- **The Genealogy of Morals**
- **Twilight of the Idols**

Each book is chunked, embedded using **Sentence Transformers**, and stored in **Qdrant** vector database for fast semantic search. The API provides endpoints for both retrieval-only queries and full RAG responses with LLM-generated answers.

---

## ✨ Features

- **Multi-Book Support** - Search across Nietzsche's complete works with book-level filtering
- **Semantic Search** - Find relevant passages using cosine similarity on embeddings
- **RAG Capabilities** - Combine retrieval with LLMs for contextual answers
- **Fast & Scalable** - Built on Qdrant vector database
- **Clean Architecture** - Separation of concerns with routes, services, and utilities

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
