from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse, JSONResponse
from dotenv import load_dotenv
from core import VectorDB
from schemas import *

load_dotenv()

vector_db = VectorDB()
# Initialize FastAPI app
app = FastAPI(title="HFAH Vector DB",redirect_slashes=False)

# Set all CORS enabled origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

@app.get("/")
def read_root():
    return PlainTextResponse(content="Healthy", status_code=200)

@app.post("/invoke", response_model=InvokeResponse)
def invoke(request: InvokeRequest):
    response = vector_db.invoke(request) 
    return response

@app.post("/add", response_model=AddResponse)
def add(request: AddRequest):
    response = vector_db.add_data(request) 
    return response

@app.get("/all")
def all():
    all_data = {
        "vectors": vector_db.vectors.tolist(),
        "metadata": vector_db.metadata
    }
    return JSONResponse(content=all_data)

# @app.post("/delete")
# def delete():
#     return PlainTextResponse(content="Delete endpoint is healthy", status_code=200)
