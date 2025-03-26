import os
import uuid
import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter
from qdrant_client import models, QdrantClient
from sentence_transformers import SentenceTransformer  

QDRANT_URL = "https://fd1f537d-a33f-4f77-9012-4c4ee4a6293e.us-east4-0.gcp.cloud.qdrant.io:6333"
QDRANT_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.h4KdLex_iotZFRwBgTFLXbzse1_U-A7nYqZIXeOMvpM"

qdrant_client = QdrantClient(url=QDRANT_URL, prefer_grpc=True, api_key=QDRANT_API_KEY, timeout=30)


def create_qdrant_collection(collection_name):
    vector_size = 384  
    distance_metric = models.Distance.COSINE  

    if qdrant_client.collection_exists(collection_name):
        qdrant_client.delete_collection(collection_name)

    qdrant_client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(size=vector_size, distance=distance_metric)
    )


def extract_text_from_pdf(pdf_file):
    try:
        with open(pdf_file, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
        return text.strip()
    except Exception as e:
        return ""


def chunk_pdf_text(pdf_file, chunk_size=1000, chunk_overlap=100):
    text = extract_text_from_pdf(pdf_file)
    
    if not text:
        return []

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter.split_text(text)


def generate_unique_id(filename):
    unique_uuid = uuid.uuid5(uuid.NAMESPACE_DNS, filename)
    return abs(unique_uuid.int) % (2**63 - 1)


def upload_chunks_to_qdrant(chunks, pdf_filename, collection_name):
    embedder = SentenceTransformer('all-MiniLM-L6-v2')  

    points = []
    for i, chunk in enumerate(chunks):
        chunk_embedding = embedder.encode(chunk).tolist()
        point_id = generate_unique_id(f"{pdf_filename}_{i}")
        
        point = models.PointStruct(
            id=point_id, 
            vector=chunk_embedding, 
            payload={
                "text": chunk,
                "filename": pdf_filename
            }
        )
        points.append(point)

    if points:
        qdrant_client.upsert(collection_name=collection_name, points=points)


def collection_create(pdf_path, collection_name):
    create_qdrant_collection(collection_name)
    pdf_filename = os.path.basename(pdf_path)
    chunks = chunk_pdf_text(pdf_path)
    upload_chunks_to_qdrant(chunks, pdf_filename, collection_name)


if __name__ == "__main__":
    pdf_path = "/Users/deepek/Desktop/rp/sample.pdf"
    collection_name = "research_papers"
    collection_create(pdf_path, collection_name)
