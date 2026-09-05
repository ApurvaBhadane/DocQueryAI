# 📄 DocQuery AI

AI-powered document question-answering system that allows users to ask questions from normal and scanned PDF documents.

## 🚀 Live Demo

https://docqueryai-yfurkvneizzsbyxd55k5mh.streamlit.app/

## ✨ Features

- 📄 PDF document upload
- 🔍 Text extraction from normal PDFs
- 🖼️ OCR support for scanned PDFs
- 🌐 English, Hindi and Marathi support
- 🧠 Multilingual semantic search
- 🔎 Hybrid semantic + lexical retrieval
- 🤖 AI-powered question answering
- 🛡️ Hallucination protection
- 📊 Retrieval confidence score
- 📑 Source page detection
- ⚡ Parallel page processing
- 💾 Document caching

## 🛠️ Tech Stack

- Python
- Streamlit
- PyMuPDF
- Tesseract OCR
- Pytesseract
- Pillow
- NumPy
- Sentence Transformers
- OpenAI / OpenRouter

## 🛡️ Hallucination Protection

The system answers questions using only the information retrieved from the uploaded document.

If the required information cannot be confidently found, DocQuery AI returns:

> Sorry, I couldn't find this information in the uploaded document.

## ⚡ Performance

- Parallel PDF page processing
- Document hash-based caching
- Embedding caching
- Session-state management

