from fastapi import FastAPI, UploadFile, File
from sentence_transformers import SentenceTransformer
from pydantic import BaseModel
import pandas as pd
import numpy as np
import uvicorn

# Load the embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

app = FastAPI()

# Endpoint to vectorize a single text
class TextData(BaseModel):
    text: str

@app.post("/vectorize-text")
async def vectorize_text(data: TextData):
    # Generate embedding
    embedding = model.encode(data.text).tolist()
    print(embedding)
    return {
        "text": data.text,
        "embedding": embedding
    }

# Endpoint to vectorize a CSV file
@app.post("/vectorize-csv")
async def vectorize_csv(file: UploadFile = File(...)):
    # Read CSV file into a DataFrame
    df = pd.read_csv(file.file)
    
    # Assuming text is in the first column
    texts = df.iloc[:, 0].tolist()
    
    # Generate embeddings for each text entry
    embeddings = model.encode(texts)
    
    # Prepare response with both text and embeddings
    results = [{"text": text, "embedding": emb.tolist()} for text, emb in zip(texts, embeddings)]
    
    return results



if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
