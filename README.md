RAG-Based Chatbot for PDFs, CSVs, and Images
  This project is a Retrieval-Augmented Generation (RAG)-based chatbot that allows users to query information from PDFs, CSVs, and images. It retrieves relevant documents using Qdrant for vector search and processes queries using Cohere and Gemini. The frontend is built using Streamlit for an interactive user experience.

Features
  Query information from pre-uploaded PDFs (10 research papers).

  Analyze data from a CSV dataset with 1,000 records.

  Upload images for processing.

  Use Cohere to refine user queries.

  Retrieve document embeddings using Qdrant.

  Generate responses using Gemini.

Interactive Streamlit frontend.

Tech Stack

Backend: Python
Vector Database: Qdrant
Query Refinement: Cohere
LLM: Gemini
Embeddings: Sentence-Transformers

Frontend: Streamlit
Setup Instructions
Prerequisites
Python 3.8+

Qdrant running locally or on the cloud

Install Dependencies
pip install -r requirements.txt
Running the Project

Start the Streamlit frontend:
streamlit run app.py
The backend processes PDFs, CSVs, and images, stores them in Qdrant, and follows a two-stage RAG approach:

Cohere refines user queries.

Qdrant retrieves relevant document chunks.

Gemini generates the final response.

Future Improvements
  Allow users to upload their own PDFs.

  Improve query response time.

  Implement semantic chunking for better retrieval.

  Support additional file types.

