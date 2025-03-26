from RAG import getTopKDocs
from query_db import getTopChunks, gemini

def getAnswer(query):
    departments = getTopKDocs(query)
    totalChunks = []
    
    for department in departments:
        topChunks = getTopChunks(query, department)
        totalChunks.extend(topChunks)
    
    finalResult = gemini(query, totalChunks)
    
    return finalResult
