from utils import *
from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse
from fastapi import File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from langchain_community.embeddings import HuggingFaceEmbeddings
import os
from fastapi import Depends
from typing import List
import openai
from dotenv import load_dotenv
import uvicorn
from fastapi import status



openai.api_key  = os.getenv("OPENAI_API_KEY")
qdrant_url  = os.getenv('qdrant_url')
qdrant_api_key  = os.getenv('qdrant_api_key')


app = FastAPI()

load_dotenv()

origins = ["*"]
app.add_middleware(
    CORSMiddleware, 
    allow_origins=origins, 
    allow_credentials=True, 
    allow_methods=["*"], 
    allow_headers=["*"]
    )

embed_fn = HuggingFaceEmbeddings(model_name='BAAI/bge-small-en-v1.5')

@app.get("/")
def startup():
    return JSONResponse("Document Reranker")

    

@app.post("/generate_vectordb")
async def generateVectordb(file: UploadFile):
    try:
        contents = await file.read()
        file_extension = file.filename[-4:].lower()
        original_filename = file.filename
        doc_list = handling_files(contents, file_extension, original_filename)
        vectordb = background_task(doc_list, embed_fn, original_filename)
        return JSONResponse(content={"message": "Response Generated Successfully!", "vectordb name": vectordb}, status_code=status.HTTP_200_OK)

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=status.HTTP_400_BAD_REQUEST)


    
@app.post("/rerank_chain")
async def retrievalLcelChain(query:str, vectordb_name: str):
    try:
        vectordb= load_local_vectordb_using_qdrant(vectordb_name, embed_fn)
        response = semantic_search_conversation(query, vectordb)
        return JSONResponse(content={"message": "Response Generated Successfully!", "Response": response}, status_code=status.HTTP_200_OK)
    except Exception as ex:
        return JSONResponse(content={"error": str(ex)}, status_code=status.HTTP_400_BAD_REQUEST)
    


@app.post("/conversational_rerank_chain")
async def conversationChainLCEL(query:str,  vectordb_name: str):
    try:
        vectordb= load_local_vectordb_using_qdrant(vectordb_name, embed_fn)
        response = conversation_retrieval_chain(query, vectordb)
        return JSONResponse(content={"message": "Response Generated Successfully!", "Response": response}, status_code=status.HTTP_200_OK)
    except Exception as ex:
        return JSONResponse(content={"error": str(ex)}, status_code=status.HTTP_400_BAD_REQUEST)



# @app.post("/conversational_chain")
# async def conversationChainLCEL(query:str, vectordb_name: str):
#     try:
#         vectordb= load_local_vectordb_using_qdrant(vectordb_name, embed_fn)
#         response = conversation_retrieval_chain(query, vectordb)
#         return JSONResponse(content={"message": "Response Generated Successfully!", "Response": response}, status_code=status.HTTP_200_OK)
#     except Exception as ex:
#         return JSONResponse(content={"error": str(ex)}, status_code=status.HTTP_400_BAD_REQUEST)


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=9393)