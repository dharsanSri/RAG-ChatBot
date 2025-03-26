import os
import uuid
import PyPDF2
import numpy as np
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

def generate_unique_id(filename):
    unique_uuid = uuid.uuid5(uuid.NAMESPACE_DNS, filename)
    return abs(unique_uuid.int) % (2**63 - 1)

def extract_text_from_pdf(pdf_path):
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
        return text.strip()
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")
        return ""

def store_research_papers_in_single_collection(folder_path, collection_name='research_papers'):
    model = SentenceTransformer('all-MiniLM-L6-v2')

    QDRANT_URL = "https://fd1f537d-a33f-4f77-9012-4c4ee4a6293e.us-east4-0.gcp.cloud.qdrant.io:6333"
    QDRANT_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.h4KdLex_iotZFRwBgTFLXbzse1_U-A7nYqZIXeOMvpM"
    client = QdrantClient(url=QDRANT_URL, prefer_grpc=True, api_key=QDRANT_API_KEY, timeout=30)

    points = []

    for filename in os.listdir(folder_path):
        if filename.endswith('.pdf'):
            pdf_path = os.path.join(folder_path, filename)
            paper_text = extract_text_from_pdf(pdf_path)

            if not paper_text:
                print(f"Skipping {filename} due to text extraction error")
                continue

            try:
                embedding = model.encode(paper_text)
            except Exception as e:
                print(f"Error embedding {filename}: {e}")
                continue

            point_id = generate_unique_id(filename)
            research_paper_name = os.path.splitext(filename)[0]

            point = PointStruct(
                id=point_id,
                vector=embedding.tolist(),
                payload={
                    'filename': filename,
                    'research_paper_name': research_paper_name,
                    'full_text': paper_text,
                    'file_path': pdf_path
                }
            )

            points.append(point)
            print(f"Prepared: {filename}")

    try:
        if points:
            client.upsert(
                collection_name=collection_name,
                points=points
            )
            print(f"Successfully stored {len(points)} research papers in collection '{collection_name}'")
        else:
            print("No papers were stored. Check your PDF files and path.")
    except Exception as e:
        print(f"Error storing points: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    research_papers_folder = '/Users/deepek/Desktop/rp'
    store_research_papers_in_single_collection(research_papers_folder)
