import streamlit as st
import numpy as np
import ollama
from pdf_reader import extract_text_from_pdf
from embeddings import create_embeddings
from rag_pipeline import create_faiss_index, search
from sentence_transformers import SentenceTransformer

# Embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

st.set_page_config(page_title="AI Study Assistant", page_icon="📚")

st.title("📚 AI Study Assistant")
st.write("Ask questions from your handwritten notes using AI")

# -------------------------
# Chat history memory
# -------------------------

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "index" not in st.session_state:
    st.session_state.index = None

if "text_chunks" not in st.session_state:
    st.session_state.text_chunks = None

# -------------------------
# Upload PDF
# -------------------------

st.header("Upload Study Notes")

uploaded_file = st.file_uploader(
    "Upload handwritten PDF",
    type=["pdf"]
)

if uploaded_file:

    with open("uploaded_file.pdf", "wb") as f:
        f.write(uploaded_file.read())

    text_chunks = extract_text_from_pdf("uploaded_file.pdf")

    embeddings = create_embeddings(text_chunks)

    index = create_faiss_index(embeddings)

    # store in session
    st.session_state.index = index
    st.session_state.text_chunks = text_chunks

    st.success("Notes indexed successfully! You can now ask questions.")

# -------------------------
# Ask question
# -------------------------

if st.session_state.index is not None:

    st.header("Ask a Question from Your Notes")

    question = st.text_input("Enter your question")

    if question:

        with st.spinner("Generating answer..."):

            query_embedding = embedding_model.encode([question])

            distances, indices = search(
                st.session_state.index,
                query_embedding
            )

            retrieved_text = ""
            pages = []

            for i in indices[0]:
                retrieved_text += st.session_state.text_chunks[i][1] + "\n"
                pages.append(st.session_state.text_chunks[i][0])

            # Create prompt for AI
            prompt = f"""
Use ONLY the notes below to answer.

Notes:
{retrieved_text}

Question:
{question}

If answer not found say:
"I don't know based on the notes."
"""

            response = ollama.chat(
                model="phi3:mini",
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )

            answer = response["message"]["content"]

            # -------------------------
            # Calculate confidence score
            # -------------------------
            # Using closest chunk distance for better confidence
            # Scale factor can be adjusted (0.0–1.0)
            closest_distance = distances[0][0]
            confidence = max(0, 100 - closest_distance * 50)  # tweak factor to make scores higher

        st.subheader("Answer")
        st.write(answer)

        st.subheader("Confidence Score")
        st.write(f"{confidence:.2f}%")

        st.subheader("Source Pages")
        st.write(sorted(list(set(pages))))

        # Save chat history
        st.session_state.chat_history.append({
            "question": question,
            "answer": answer,
            "confidence": confidence,
            "pages": sorted(list(set(pages)))
        })

# -------------------------
# Display chat history
# -------------------------

if st.session_state.chat_history:

    st.header("Chat History")

    for chat in reversed(st.session_state.chat_history):

        st.markdown("**Question:**")
        st.write(chat["question"])

        st.markdown("**Answer:**")
        st.write(chat["answer"])

        st.markdown("**Confidence Score:**")
        st.write(f"{chat['confidence']:.2f}%")

        st.markdown("**Source Pages:**")
        st.write(chat['pages'])

        st.markdown("---")
