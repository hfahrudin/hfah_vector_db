
from pydantic import BaseModel
from typing import Optional, List

class AddRequest(BaseModel):
    text: str                 # text to embed
    label: str                # required label
    source: str = None        # optional

class AddResponse(BaseModel):
    status: str
    vector_index: int

class InvokeRequest(BaseModel):
    query: str               # text to search for
    top_k: int = 5           # number of results to return

class ResultItem(BaseModel):
    index: int               # vector index in DB
    score: float             # similarity score
    metadata: dict           # stored metadata

class InvokeResponse(BaseModel):
    results: List[ResultItem]