from chunkpapers import collection_create
import os

folder_path = "/Users/deepek/Desktop/rp"
collection_name = "researchpaper-"

def create_collections():
    i = 2
    for pdf_file in os.listdir(folder_path):
        col_name = f"{collection_name}{i}"
        collection_create(os.path.join(folder_path, pdf_file), col_name)
        i += 1

create_collections()
