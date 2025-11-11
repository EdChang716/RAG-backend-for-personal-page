# RAG-backend-for-personal-page

An end-to-end **RAG (Retrieval-Augmented Generation)** assistant that allows visitors or recruiters to ask questions about my **background, research, and projects** directly on my personal website — and receive concise, evidence-grounded answers.

---

## Overview

This project integrates OpenAI embeddings, Supabase (pgvector), and a serverless Vercel API to enable intelligent Q&A over my personal documents such as resumes, research papers, and project descriptions.  

It powers a custom chat widget embedded on my GitHub Pages portfolio, designed to respond naturally to both **professional inquiries** (“What machine learning projects have you worked on?”) and **casual questions** (“Tell me about your background.”).

---

## Architecture

frontend (GitHub Pages)
│
▼
Vercel serverless API (/api/search, /api/generate)
│
▼
Supabase (pgvector) ← embeddings generated via OpenAI API
│
▼
data ingestion (PDF, Markdown, HTML → text chunks)

---

## Repository Structure

| Folder / File | Description |
|----------------|-------------|
| **api/** | Serverless API routes deployed on Vercel. Includes:<br>• `search.js` – vector search endpoint (semantic retrieval via Supabase RPC)<br>• `generate.js` – generation endpoint that composes final LLM responses<br>• `health.js` – health check for deployment monitoring |
| **lib/** | Core RAG logic and utilities:<br>• `embeddings.js` – OpenAI embeddings generation<br>• `retrieve.js` – semantic retrieval from pgvector<br>• `intent.js` – simple intent detection and context routing (e.g., STAR, project, small talk)<br>• `supabase.js` – Supabase client initialization and RPC helpers |
| **data/** | Stores processed documents and metadata used for local testing or ingestion. |
| **ingest.py** | One-time ingestion pipeline that parses documents (PDF/Markdown/HTML), chunks text, and uploads embeddings + metadata to Supabase. |
| **package.json** | Node dependencies and script definitions. |
| **README.md** | Project documentation (this file). |

---

## Features

### 🔹 Ingestion & Indexing
- Automatically parses structured and unstructured documents (PDF, Markdown, HTML).
- Generates **OpenAI embeddings** and stores vectors in **Supabase pgvector**.
- Uses a SQL **RPC function** for efficient semantic search with cosine similarity.

### 🔹 Retrieval Quality Enhancements
- Query rewriting based on chat history for contextual continuity.  
- Lightweight **query expansion** and **LLM-based re-ranking** of retrieved chunks.  
- Intent-aware prompting (e.g., uses STAR-style summarization for experience-related questions).

### 🔹 Generation
- Grounded answer synthesis — no hallucination or fake citations.  
- Graceful fallback when retrieval is low-confidence.  
- Context-aware tone adaptation for professional or conversational queries.

### 🔹 Frontend & Deployment
- Embedded **custom chat widget** on GitHub Pages.  
- **Vercel serverless APIs** with strict CORS and error handling.  
- Handles both warm starts and cold starts with caching optimizations.

---

## Example Queries

| Question | Example Response |
|-----------|------------------|
| “Tell me about your LSTM water quality prediction project.” | Summarizes full pipeline (MICE-RF + LSTM-ED framework) and points to GitHub repo. |
| “What’s your experience with NLP?” | Mentions research on text embeddings and semantic similarity, with context from publication data. |
| “Can you explain your internship experience?” | Generates a STAR-structured summary with verified details from the resume. |

---

## Tech Stack

| Component | Technology |
|------------|-------------|
| Embeddings | OpenAI text-embedding-3-large |
| Vector Store | Supabase (PostgreSQL + pgvector) |
| Backend | Node.js / Vercel serverless functions |
| Data Ingestion | Python (pypdf + unstructured + OpenAI SDK) |
| Frontend | Static site (GitHub Pages + custom JS widget) |
| Auth & CORS | Vercel environment variables + Supabase service role key |

---

## Setup

### Install dependencies
```bash
npm install
```
### Set up environment variables
Create a .env.local file and configure:
```bash
OPENAI_API_KEY=sk-xxxx
SUPABASE_URL=https://xxxx.supabase.co
SUPABASE_SERVICE_ROLE=xxxx
```
### Run ingestion
```bash
python ingest.py
```
### Deploy API on Vercel
```bash
vercel deploy --prod
```
### Embed chatbot on personal site
Add your chat widget script (or iframe) to the portfolio HTML page, pointing to /api/search and /api/generate.

## Outcome
This chatbot enables recruiters or visitors to:

Ask targeted, context-aware questions about my work.

Receive concise and accurate summaries with verified evidence.

Navigate to related documents (Resume / Projects / GitHub) when relevant.

It demonstrates end-to-end mastery of RAG pipelines, embedding search, and LLM orchestration, integrated into a production-ready portfolio application.


