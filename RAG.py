import os
import pathlib
import pdfplumber
from itertools import chain
from operator import itemgetter
from typing import List

import fitz
import time

from langchain.schema import Document
from langchain.load import dumps, loads
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_core.documents.base import Document

from sentence_transformers import SentenceTransformer
from langchain_community.llms import Cohere
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain_cohere import CohereEmbeddings

from qdrant_client import models, QdrantClient


TOP_K = 5
MAX_DOCS_FOR_CONTEXT = 10
QDRANT_URL = "https://fd1f537d-a33f-4f77-9012-4c4ee4a6293e.us-east4-0.gcp.cloud.qdrant.io:6333"
QDRANT_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.h4KdLex_iotZFRwBgTFLXbzse1_U-A7nYqZIXeOMvpM"
QDRANT_COLLECTION_NAME = "department-wise"

COHERE_API_KEY = "lb5TT3QgjdHf8yqcIxoFIXtFc5pysCxS2EmfUFFj"
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

qdrant_client = QdrantClient(url=QDRANT_URL, prefer_grpc=True, api_key=QDRANT_API_KEY)

def read_pdf_files(directory: str) -> List[Document]:
    documents = []
    for pdf_file in pathlib.Path(directory).glob('*.pdf'):
        with pdfplumber.open(pdf_file) as pdf:
            text = "".join(page.extract_text() or "" for page in pdf.pages)
            metadata = {"filename": pdf_file.stem}
            documents.append(Document(page_content=text, metadata=metadata))
    return documents

def upload_chunks_to_qdrant(documents):
    records_to_upload = []
    
    contents = [doc.page_content for doc in documents]
    embeddings = embedding_model.encode(contents)

    for idx, (content, vector) in enumerate(zip(contents, embeddings)):
        record = models.PointStruct(
            id=idx,
            vector=vector,
            payload={"id": idx, "page_content": content}
        )
        records_to_upload.append(record)

    qdrant_client.upload_points(
        collection_name=QDRANT_COLLECTION_NAME,
        points=records_to_upload
    )

def query_qdrant(query: str, collection_name: str, top_k: int) -> List[dict]:
    query_embedding = embedding_model.encode(query).tolist()
    results = qdrant_client.search(
        collection_name=collection_name,
        query_vector=query_embedding,
        limit=top_k,
        with_payload=True
    )
    return results

def reciprocal_rank_fusion(all_results: List[List[dict]], collection_name: str, k: int) -> List[str]:
    rrf_scores = {}
    for query_results in all_results:
        for rank, doc in enumerate(query_results):
            doc_id = doc.id
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (rank + 1)
    
    sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    top_k_docs = sorted_docs[:k]
    
    departments = []
    for doc_id, _ in top_k_docs:
        records = qdrant_client.retrieve(
            collection_name=collection_name,
            ids=[doc_id],
            with_payload=True
        )
        if records:
            departments.append(records[0].payload.get('department'))
    
    return departments

def build_rrf_chain(user_query: str, collection_name: str, k: int) -> List[str]:
    query_gen_lambda = RunnableLambda(lambda x: query_generator(x))
    query_qdrant_lambda = RunnableLambda(lambda queries: [query_qdrant(q, collection_name, k) for q in queries])
    rrf_lambda = RunnableLambda(lambda results: reciprocal_rank_fusion(results, collection_name, k))
    
    chain = (
        RunnableLambda(lambda x: {"query": x})  
        | query_gen_lambda
        | query_qdrant_lambda
        | rrf_lambda
    )
    
    return chain.invoke(user_query)

def query_generator(original_query: dict) -> List[str]:
    query = original_query.get("query")

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant that generates multiple search queries based on a single input query."),
        ("human", "Generate multiple search queries related to: {original_query}. When creating queries, please refine or add closely related contextual information, without significantly altering the original query's meaning.\n\nOUTPUT (3 queries):")
    ])

    model = Cohere(cohere_api_key=COHERE_API_KEY)
    query_generator_chain = (
        prompt | model | StrOutputParser() | (lambda x: [q.strip() for q in x.split("\n") if q.strip()])
    )

    queries = query_generator_chain.invoke({"original_query": query})
    queries.insert(0, query)

    return queries

def getTopKDocs(query):
    return build_rrf_chain(query, QDRANT_COLLECTION_NAME, 2)
