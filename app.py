import streamlit as st
import os
from testpq import getFinalAnswer
from processQeury import getAnswer
from image_query import Answer
from image_processing import extract_text_from_image
from embedder import generate_embeddings
from storage import store_in_qdrant


def initialize_chat_history():
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hi! Choose CSV, PDF, or Image Chatbot to get started."}
        ]


def display_chat_history():
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


def csv_chatbot():
    st.title("📊 CSV Chatbot")
    display_chat_history()
    
    if prompt := st.chat_input("Ask me about the CSV..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("Processing CSV..."):
                answer = getAnswer(prompt)
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})


def pdf_chatbot():
    st.title("📄 PDF Chatbot")
    display_chat_history()
    
    if prompt := st.chat_input("Ask me about the PDF..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("Processing PDF..."):
                answer = getFinalAnswer(prompt)
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})


def image_chatbot():
    st.title("🖼️ Image Chatbot")
    display_chat_history()

    uploaded_image = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
    
    if uploaded_image:
        with st.spinner("Extracting text from image..."):
            extracted_text = extract_text_from_image(uploaded_image)
            print(f"Extracted Text: {extracted_text}")

            file_name, _ = os.path.splitext(uploaded_image.name)
            embedding = generate_embeddings(extracted_text)
            store_in_qdrant(extracted_text, embedding, file_name)

            st.session_state.messages.append(
                {"role": "assistant", "content": f"Extracted Text: {extracted_text}"}
            )
            st.success("Text extracted and stored successfully!")

    query = st.chat_input("Ask me a question about the extracted text...")

    if query:
        st.session_state.messages.append({"role": "user", "content": query})
        
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            with st.spinner("Searching for answers..."):
                response = Answer(query, file_name)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})


def main():
    st.set_page_config(page_title="File-Based RAG Chatbot", page_icon=":robot:", layout="wide")
    initialize_chat_history()

    st.title("Welcome to File-Based RAG Chatbot")
    st.write("Choose an option below to continue:")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("📊 CSV Chatbot"):
            st.session_state.chat_mode = "csv"

    with col2:
        if st.button("📄 PDF Chatbot"):
            st.session_state.chat_mode = "pdf"

    with col3:
        if st.button("🖼️ Image Chatbot"):
            st.session_state.chat_mode = "image"
    
    if "chat_mode" in st.session_state:
        if st.session_state.chat_mode == "csv":
            csv_chatbot()
        elif st.session_state.chat_mode == "pdf":
            pdf_chatbot()
        elif st.session_state.chat_mode == "image":
            image_chatbot()


if __name__ == "__main__":
    main()
