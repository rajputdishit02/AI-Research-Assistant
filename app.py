import os
import traceback
import streamlit as st
import ollama

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings


st.set_page_config(page_title="AI Research Assistant", page_icon="📚")

TEMP_FOLDER = "temp_files"
os.makedirs(TEMP_FOLDER, exist_ok=True)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "document_summaries" not in st.session_state:
    st.session_state.document_summaries = []

if "key_information" not in st.session_state:
    st.session_state.key_information = []


with st.sidebar:
    st.header("📚 AI Research Assistant")
    st.write("Upload PDFs, ask questions, summarize documents, and extract key information.")
    st.write("Built with LangChain, FAISS, Sentence Transformers and Ollama.")


st.title("📚 AI Research Assistant")
st.write("Upload PDFs and analyze them using local RAG and Ollama.")


uploaded_files = st.file_uploader(
    "Upload one or more PDFs",
    type="pdf",
    accept_multiple_files=True
)


if uploaded_files:
    try:
        with st.spinner("Processing PDFs..."):
            all_documents = []
            st.session_state.uploaded_file_names = [
                file.name for file in uploaded_files
            ]

            for uploaded_file in uploaded_files:
                safe_file_name = uploaded_file.name.replace(" ", "_")
                temp_path = os.path.join(TEMP_FOLDER, f"temp_{safe_file_name}")

                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.read())

                loader = PyPDFLoader(temp_path)
                documents = loader.load()

                for doc in documents:
                    doc.metadata["source_file"] = uploaded_file.name

                all_documents.extend(documents)

            st.session_state.documents = all_documents

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=700,
                chunk_overlap=150
            )

            chunks = splitter.split_documents(all_documents)

            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-mpnet-base-v2"
            )

            vectorstore = FAISS.from_documents(chunks, embeddings)
            st.session_state.vectorstore = vectorstore

        st.success(f"{len(uploaded_files)} PDF(s) processed successfully!")

    except Exception as e:
        st.error("Error processing uploaded PDFs.")
        st.exception(e)
        st.text(traceback.format_exc())


if uploaded_files and "documents" in st.session_state:
    col1, col2 = st.columns(2)

    with col1:
        summarize_clicked = st.button("📄 Summarize All Documents")

    with col2:
        extract_clicked = st.button("📋 Extract Key Information")

    if summarize_clicked:
        try:
            with st.spinner("Generating document summaries..."):
                summaries = []

                for file_name in st.session_state.uploaded_file_names:
                    file_docs = [
                        doc
                        for doc in st.session_state.documents
                        if doc.metadata.get("source_file") == file_name
                    ]

                    summary_context = "\n\n".join(
                        [doc.page_content for doc in file_docs[:3]]
                    )

                    summary_prompt = f"""
You are an expert document analyst.

Analyze the document content and provide:

1. Document Type
2. Main Purpose
3. Key Points in 3-5 bullet points
4. Short Summary

Keep the answer clear and concise.

Document Name:
{file_name}

Document Content:
{summary_context}
"""

                    response = ollama.chat(
                        model="llama3.2:3b",
                        messages=[{"role": "user", "content": summary_prompt}]
                    )

                    summaries.append(
                        {
                            "file": file_name,
                            "summary": response["message"]["content"]
                        }
                    )

                st.session_state.document_summaries = summaries

        except Exception as e:
            st.error("Error generating document summaries.")
            st.exception(e)
            st.text(traceback.format_exc())

    if extract_clicked:
        try:
            with st.spinner("Extracting key information..."):
                extracted_items = []

                for file_name in st.session_state.uploaded_file_names:
                    file_docs = [
                        doc
                        for doc in st.session_state.documents
                        if doc.metadata.get("source_file") == file_name
                    ]

                    extraction_context = "\n\n".join(
                        [doc.page_content for doc in file_docs[:4]]
                    )

                    extraction_prompt = f"""
You are an expert information extraction assistant.

Extract the most important structured information from the document.

Return the answer in this format:

Document Type:
Main Topic:
Important Names:
Important Dates:
Important Amounts or Numbers:
Key Identifiers:
Key Requirements or Conditions:
Short Notes:

If a field is not available, write "Not found".

Document Name:
{file_name}

Document Content:
{extraction_context}
"""

                    response = ollama.chat(
                        model="llama3.2:3b",
                        messages=[{"role": "user", "content": extraction_prompt}]
                    )

                    extracted_items.append(
                        {
                            "file": file_name,
                            "info": response["message"]["content"]
                        }
                    )

                st.session_state.key_information = extracted_items

        except Exception as e:
            st.error("Error extracting key information.")
            st.exception(e)
            st.text(traceback.format_exc())


if st.session_state.document_summaries:
    st.subheader("📄 Document Summaries")

    for item in st.session_state.document_summaries:
        with st.expander(item["file"]):
            st.write(item["summary"])


if st.session_state.key_information:
    st.subheader("📋 Extracted Key Information")

    for item in st.session_state.key_information:
        with st.expander(item["file"]):
            st.write(item["info"])


st.divider()

question = st.text_input("Ask a question about the uploaded PDFs")


if question and "vectorstore" in st.session_state:
    try:
        with st.spinner("Generating answer..."):
            context_docs = []

            for file_name in st.session_state.uploaded_file_names:
                docs_for_file = st.session_state.vectorstore.similarity_search(
                    question,
                    k=3,
                    filter={"source_file": file_name}
                )
                context_docs.extend(docs_for_file)

            context = "\n\n".join(
                [
                    f"File: {doc.metadata.get('source_file', 'Unknown file')}\n"
                    f"Page: {doc.metadata.get('page', 'Unknown') + 1 if doc.metadata.get('page', 'Unknown') != 'Unknown' else 'Unknown'}\n"
                    f"Content:\n{doc.page_content}"
                    for doc in context_docs
                ]
            )

            prompt = f"""
You are an AI assistant helping users understand PDF documents.

Use only the provided document context to answer the question.
If the context contains the answer, give it clearly.
If the answer is not present in the context, say:
"I could not find this information in the uploaded documents."

Do not make up information.
Do not include long raw document text.
Mention the file name and page number when useful.

Context:
{context}

Question:
{question}

Answer:
"""

            response = ollama.chat(
                model="llama3.2:3b",
                messages=[{"role": "user", "content": prompt}]
            )

            answer = response["message"]["content"]

        st.subheader("Answer")
        st.write(answer)

        st.session_state.chat_history.append(
            {
                "question": question,
                "answer": answer
            }
        )

        unique_sources = []

        for doc in context_docs:
            source_file = doc.metadata.get("source_file", "Unknown file")
            page = doc.metadata.get("page", "Unknown")
            page_display = page + 1 if page != "Unknown" else "Unknown"

            source = (source_file, page_display)

            if source not in unique_sources:
                unique_sources.append(source)

        st.subheader("Sources")
        for source_file, page in unique_sources:
            st.write(f"{source_file} — Page {page}")

        with st.expander("View retrieved context"):
            st.text(context[:5000])

    except Exception as e:
        st.error("Error generating answer.")
        st.exception(e)
        st.text(traceback.format_exc())


if st.session_state.chat_history:
    st.divider()
    st.subheader("Chat History")

    for i, chat in enumerate(reversed(st.session_state.chat_history), start=1):
        with st.expander(
            f"Question {len(st.session_state.chat_history) - i + 1}"
        ):
            st.markdown(f"**Question:** {chat['question']}")
            st.markdown(f"**Answer:** {chat['answer']}")


st.divider()
st.caption("Built with LangChain, FAISS, Sentence Transformers and Ollama")