from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from dotenv import load_dotenv
from core import VectorDB

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


@app.post("/invoke")
def invoke():
    return PlainTextResponse(content="Invoke endpoint is healthy", status_code=200)

@app.post("/add")
def add():
    return PlainTextResponse(content="Add endpoint is healthy", status_code=200)

@app.post("/delete")
def delete():
    return PlainTextResponse(content="Delete endpoint is healthy", status_code=200)
