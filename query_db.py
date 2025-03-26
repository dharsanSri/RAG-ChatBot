from sentence_transformers import SentenceTransformer
from qdrant_client import models, QdrantClient
import google.generativeai as genai

QDRANT_URL = "https://fd1f537d-a33f-4f77-9012-4c4ee4a6293e.us-east4-0.gcp.cloud.qdrant.io:6333"
QDRANT_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.h4KdLex_iotZFRwBgTFLXbzse1_U-A7nYqZIXeOMvpM"
QDRANT_COLLECTION_NAME = "research-papers-chunk-2"

GEMINI_API_KEY = "AIzaSyB1bWIahQuwnNbTMQOygJVPjx4Tu3WFyy8"
genai.configure(api_key=GEMINI_API_KEY)

model = genai.GenerativeModel("gemini-2.0-flash-thinking-exp-01-21")

qdrant_client = QdrantClient(url=QDRANT_URL, prefer_grpc=True, api_key=QDRANT_API_KEY, timeout=30)

embedder = SentenceTransformer('all-MiniLM-L6-v2')

def vector_search(query, collection_name, top_k=8):
    query_vector = embedder.encode(query).tolist()
    
    search_result = qdrant_client.search(
        collection_name=collection_name,
        query_vector=query_vector,
        limit=top_k
    )

    results = []
    for result in search_result:
        chunk_text = result.payload.get('text', 'No text found')
        results.append(chunk_text)
    
    return results

def gemini(query, chunks):
    context = "\n".join([f"{i+1}. {chunk}" for i, chunk in enumerate(chunks)])
    
    prompt = f"""
    You are a highly knowledgeable assistant. Based on the given context, please provide a well-crafted answer to the query below. Use the provided information from the context as reference material.

    ### Context:
    {context}

    ### Query:
    {query}

    Provide a concise, clear, and informative response as paragraphs of text based on the query.
    """

    response = model.generate_content(prompt)

    if response.candidates and len(response.candidates) > 0:
        return response.text
    else:
        return "No valid content was returned. Please adjust your prompt or try again."

def geminiWithReferences(query, chunks, docIds):
    context = "\n".join([f"{i+1}. {chunk}" for i, chunk in enumerate(chunks)])
    ids = ' '.join(str(num) for num in docIds)
    
    prompt = f"""
    You are a highly knowledgeable assistant. Based on the given context, please provide a well-crafted answer to the query below. Use the provided information from the context as reference material.

    ### Context:
    {context}

    ### Query:
    {query}

    ### Relevant Document IDs
    {ids}

    The relevant document ids from which the context was taken from is given. The id to paper title mapping is also given.
    Provide a concise, clear, and informative response as paragraphs of text based on the query. Also, take the corresponding paper title
    from the provided Research Paper to ID map by referring the Relevant document Ids and put them as references at the end of the answer.
    Put all paper names from the Relevant Document IDs.
    """

    response = model.generate_content(prompt)

    if response.candidates and len(response.candidates) > 0:
        return response.text
    else:
        return "No valid content was returned. Please adjust your prompt or try again."

def getTopChunks(query, collectionName):
    top_chunks = vector_search(query, collectionName, top_k=3)
    return top_chunks
