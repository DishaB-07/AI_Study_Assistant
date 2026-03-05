# AI Study Assistant 📚

**AI Study Assistant** is an intelligent tool that helps students and learners ask questions directly from their handwritten notes. Using AI, embeddings, and a retrieval-augmented generation (RAG) pipeline, it retrieves relevant content from your notes and provides answers along with a confidence score.

---

## Features

- **Smart Q&A** – Ask questions and get precise answers based solely on your notes.  
- **Confidence Score** – Shows how confident the AI is in the answer.  
- **Source Pages** – Displays which pages of the notes the answer was retrieved from.  
- **Topic Summarization** – Automatically extracts and indexes key information from PDFs.  
- **Interactive Chat** – Maintains chat history for easy review.  
- **Supports Handwritten Notes** – Works with scanned PDFs of handwritten study notes.

---

## Tech Stack

- **Frontend:** Streamlit  
- **Backend / AI:** Python, Ollama API, SentenceTransformers  
- **RAG (Retrieval-Augmented Generation):** FAISS + embeddings  
- **PDF Processing:** PyPDF2 or similar library  

---

## Deployment
## 🔗 Live Project Link

You can access the live project here: [https://aistudyassistant.streamlit.app/](https://aistudyassistant.streamlit.app/)


To run locally:

pip install -r requirements.txt


streamlit run app.py
