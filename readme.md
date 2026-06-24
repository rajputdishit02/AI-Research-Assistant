# AI Research Assistant

An AI-powered document assistant that allows users to upload PDFs, ask questions, summarize documents, and extract key information using local RAG.

## Features

- Upload one or more PDF documents
- Ask questions about uploaded documents
- Generate document summaries
- Extract key information from PDFs
- View source pages used for answers
- Maintain chat history
- Runs locally using Ollama

## Tech Stack

- Python
- Streamlit
- LangChain
- FAISS
- Sentence Transformers
- Ollama
- Llama 3.2

## How It Works

1. PDFs are uploaded through Streamlit.
2. Text is extracted using PyPDFLoader.
3. Text is split into chunks.
4. Embeddings are created using Sentence Transformers.
5. FAISS stores the document vectors.
6. Relevant chunks are retrieved for user questions.
7. Ollama generates answers using the retrieved context.

## Installation

```bash
pip install -r requirements.txt

## Screenshots

## Upload and Process PDFs
Screenshot/upload_screen.png

## Question Answering
Screenshot/question_answering.png

## Document Summaries
readme.md Screenshot/document_summary.png

## Extract Key Information
Screenshot/key_information.png
