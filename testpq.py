from testrag import getTopKDocs
from query_db import getTopChunks, gemini, geminiWithReferences

def getFinalAnswer(query):
    """Get Final Query Response

    Args:
        query (str): input query

    Returns:
        finalResult: consolidated answer to the query 
    """
    doc_ids = getTopKDocs(query)
    print(doc_ids)
    
    totalChunks = []
    
    for doc_id in doc_ids:
        topChunks = getTopChunks(query, doc_id)
        totalChunks.extend(topChunks)
    
    finalResult = gemini(query, totalChunks)
    return finalResult


