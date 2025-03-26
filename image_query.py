from query_db import getTopChunks, gemini

def Answer(query, file_name):
    totalChunks = []
    topChunks = getTopChunks(query, file_name)
    totalChunks.extend(topChunks)
    
    finalResult = gemini(query, totalChunks)
  
    return finalResult
