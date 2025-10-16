
## HFAH Vector DB

Simple vector database API using FastAPI and MiniLM embeddings.

### Setup and Run

#### 1. Create a virtual environment
```bash
python -m venv venv
```
#### 2. Activate the virtual environment

* **Windows:**

```bash
venv\Scripts\activate
```

* **Linux / macOS:**

```bash
source venv/bin/activate
```

#### 3. Install dependencies

```bash
pip install -r requirements.txt
```

#### 4. Run the FastAPI app

```bash
uvicorn app:app --reload --port 8000
```

#### 5. Access

* API: `http://127.0.0.1:8000`
* Docs: `http://127.0.0.1:8000/docs`

