# RAG-Based Chatbot for PDFs, CSVs, and Images

This project is a **Retrieval-Augmented Generation (RAG)-based chatbot** that allows users to query information from **PDFs, CSVs, and images**. The system retrieves relevant documents using **Qdrant** for vector search and processes queries using **Gemini and Cohere**. The frontend is built using **Streamlit**.

## Features

- **PDF Processing**: A set of 10 research papers has been pre-uploaded for querying.
- **CSV Data Processing**: A structured dataset with 1,000 records is available for analysis.
- **Image Uploading**: Users can upload images for processing.
- **Query Generation**: Uses **Cohere** to generate refined search queries.
- **Vector Search**: Stores and retrieves document embeddings using **Qdrant**.
- **LLM Response Generation**: Uses **Gemini** for generating answers based on retrieved documents.
- **Frontend**: Implemented using **Streamlit** for an interactive user interface.

## Tech Stack

- **Python** – Backend logic and data processing
- **Qdrant** – Vector database for efficient search
- **Cohere** – Used for generating multiple search queries
- **Gemini** – LLM for generating final responses
- **Streamlit** – Web-based user interface
- **Sentence-Transformers** – For text embeddings

## Setup Instructions

### Prerequisites

- Python **3.8+**
- Qdrant running locally or on the cloud

### Install Dependencies

```sh
pip install -r requirements.txt

```sh
streamlit run app.py

