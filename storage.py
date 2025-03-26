from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, Distance, VectorParams
import uuid

QDRANT_URL = "https://fd1f537d-a33f-4f77-9012-4c4ee4a6293e.us-east4-0.gcp.cloud.qdrant.io:6333"
QDRANT_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.h4KdLex_iotZFRwBgTFLXbzse1_U-A7nYqZIXeOMvpM"

client = QdrantClient(url=QDRANT_URL, prefer_grpc=True, api_key=QDRANT_API_KEY, timeout=30)

COLLECTION_NAME = "image_text_data"

def ensure_collection(collect):
    """Create a collection in Qdrant if it doesn't exist."""
    collections = client.get_collections()
    if collect not in [col.name for col in collections.collections]:
        client.create_collection(
            collect,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE)
        )

def store_in_qdrant(text, embedding, cl):
    """Store extracted text and its embeddings in Qdrant."""
    ensure_collection(cl)
    point_id = str(uuid.uuid4())  
    client.upsert(
        collection_name=cl,
        points=[
            PointStruct(id=point_id, vector=embedding, payload={"text": text})
        ]
    )
    print(f"Stored in Qdrant: {text[:50]}...")
