# Cinesearch AI

**Cinesearch AI** is a small Retrieval-Augmented-Generation (RAG) movie search service built with FastAPI that uses semantic search over the IMDB Top 1000 dataset (stored in Qdrant) and a text-generation model (Google Gemini via `google.generativeai`) to produce natural-language explanations and summaries for user queries.

The project consists of two main parts:

1. An analysis / embedding Colab notebook (`IMDB_Data_Analysis_and_Vector_Embeddings.ipynb`) used to explore the IMDB Top 1000 dataset, compute embeddings with `sentence-transformers`, and upsert vectors + metadata into a Qdrant collection named `IMDB`.
2. A FastAPI backend (`main.py` or similar — the code snippet you provided) that exposes a small web UI (Jinja2 templates) and a `/search/` endpoint which performs semantic search against Qdrant and calls Gemini to produce a human-friendly summary.

---

## Key technologies

* Python 3.8+
* FastAPI
* Uvicorn (ASGI server)
* Sentence Transformers (`sentence-transformers`, model: `all-MiniLM-L6-v2`)
* Qdrant (`qdrant-client`) as the vector database
* Google Generative AI SDK (`google.generativeai`) for text generation (Gemini)
* Jinja2 templates for optional frontend
* `python-dotenv` for local environment variable management

---

## Project structure (suggested)

```
Cinesearch-AI/
├─ README.md
├─ requirements.txt
├─ main.py                  # your FastAPI app (code you shared)
├─ templates/
│  └─ index.html            # simple search UI
├─ notebooks/
│  └─ IMDB_Data_Analysis_and_Vector_Embeddings.ipynb
├─ scripts/
│  └─ ingest_imdb_to_qdrant.py  # helper to create collection & upsert vectors
└─ .env                     # local environment variables (not committed)
```

---

## Getting started — install & prepare

1. Clone the repo (or copy files to a project folder).

2. Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate    # (Linux / macOS)
.venv\Scripts\activate     # (Windows)
pip install --upgrade pip
pip install -r requirements.txt
```

**Example `requirements.txt`** (create this file if you don't have one):

```
fastapi
uvicorn[standard]
sentence-transformers
qdrant-client
google-generativeai
python-dotenv
jinja2
pydantic
```

> Note: `sentence-transformers` will pull `transformers` & model weights when you run the code for the first time.

---

## Environment variables

Create a `.env` file in the project root (do *not* commit this file). The FastAPI app expects the following environment variables:

```
QDRANT_URL=<your_qdrant_url>            # e.g. http://localhost:6333 or hosted qdrant endpoint
QDRANT_API_KEY=<your_qdrant_api_key>    # optional for local installs
GOOGLE_API_KEY=<your_google_api_key>    # API key for google.generativeai
```

If you run Qdrant locally (docker), `QDRANT_URL` often looks like `http://localhost:6333` and `QDRANT_API_KEY` can be omitted unless you've configured one.

---

## How the FastAPI app works (summary of your code)

* On startup the app:

  * Loads environment variables with `python-dotenv`.
  * Loads a SentenceTransformer model once: `All-MiniLM-L6-v2`.
  * Initializes a `QdrantClient` using `QDRANT_URL` and optional `QDRANT_API_KEY`.
  * Configures Google Generative AI with `GOOGLE_API_KEY`.

* `/` (GET) serves a Jinja2-rendered `index.html` template for a small UI.

* `/search/` (POST) accepts a JSON body: `{ "query": "..." }`.

  1. The query text is converted to an embedding via `model.encode(query)`.
  2. This vector is used to perform a semantic search in Qdrant (collection: `IMDB`) with `limit=5`.
  3. If no results returned, the endpoint responds accordingly.
  4. For each returned hit, the app extracts useful metadata from `payload` (Title, Description, Genre, Votes, Rate, Year, Director, Stars, Metascore, Duration, Gross).
  5. The selected metadata is concatenated into a text `context` used to craft a prompt for Gemini.
  6. The prompt asks Gemini to produce a clean textual summary (no bullets or star ratings — as requested in the prompt).
  7. The generated text is returned together with the raw `results` payloads.

---

## Ingesting the IMDB dataset into Qdrant

You should already have a notebook `IMDB_Data_Analysis_and_Vector_Embeddings.ipynb` which:

* Loads the IMDB Top 1000 dataset (CSV)
* Performs basic EDA (distribution of ratings, years, runtime, genres, etc.)
* Creates embeddings using `SentenceTransformer('all-MiniLM-L6-v2')`
* Creates a Qdrant collection named `IMDB` with the correct `vector_size` (the embedding dimension from the model — `384` for `all-MiniLM-L6-v2`) and suitable distance metric (COSINE or DOT)
* Upserts points where each point contains:

  * `id` (unique, e.g. row index or IMDB id)
  * `vector` (the embedding as a list)
  * `payload` (a dict with Title, Description, Genre, Votes, Rate, Year, Director, Stars, Metascore, Duration, Gross, etc.)

If you don't have a helper ingestion script, here's an example you can adapt and run locally or in Colab:

```python
# scripts/ingest_imdb_to_qdrant.py
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance, PointStruct
import pandas as pd
import os

model = SentenceTransformer('all-MiniLM-L6-v2')
qdrant = QdrantClient(url=os.getenv('QDRANT_URL'), api_key=os.getenv('QDRANT_API_KEY'))

# Load CSV
df = pd.read_csv('IMDB_top_1000.csv')

# Prepare collection
vector_size = 384
qdrant.recreate_collection(collection_name='IMDB', vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE))

# Build points in batches
points = []
for idx, row in df.iterrows():
    emb = model.encode(row['Description']).tolist()
    payload = {
        'Title': row['Title'],
        'Description': row['Description'],
        'Genre': row.get('Genre'),
        'Votes': row.get('Votes'),
        'Rate': row.get('Rate'),
        'Year': row.get('Year'),
        'Director': row.get('Director'),
        'Stars': row.get('Stars'),
        'Metascore': row.get('Metascore'),
        'Duration': row.get('Duration'),
        'Gross': row.get('Gross')
    }
    points.append(PointStruct(id=idx, vector=emb, payload=payload))

# Upload in batches (example batch size = 128)
from more_itertools import chunked
for batch in chunked(points, 128):
    qdrant.upsert(collection_name='IMDB', points=batch)
```

> Make sure the `vector_size` matches the embedding model (`all-MiniLM-L6-v2` → 384 dims).

---

## Example usage — API request

### Using `curl` (JSON POST)

```bash
curl -X POST "http://localhost:8000/search/" \
  -H "Content-Type: application/json" \
  -d '{"query": "feel-good family movies with great cinematography"}'
```

**Response (example)**

```json
{
  "results": [
    { "Title": "...", "Description": "...", "Genre": "...", ... },
    ...
  ],
  "summary": "Gemini-generated summary text"
}
```

---

## Running the app locally

Start the FastAPI app with Uvicorn:

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000` in your browser (the `index.html` template should render).

---

## Frontend

The project uses a Jinja2 `index.html` in `templates/` for a simple UI. The UI should `POST` the query to `/search/` and display both the raw results and the generated summary. You can implement a small JavaScript `fetch()` call on the page to communicate with the backend.

---

## Important notes & troubleshooting

* **No results from Qdrant:** Confirm the `IMDB` collection exists and contains points. Check that you used the same vector size and distance metric when creating the collection.
* **Embedding dimension mismatch:** If you change the embedding model, update `vector_size` in collection creation.
* **Rate limits / API keys:** Ensure your Google API key is valid and has access to the Generative AI product. Handle exceptions for rate-limiting.
* **Prompt design:** The current prompt instructs Gemini to return a short summary without bullet points or stars. Tweak the prompt for style, length, or tone.
* **Security:** Keep `.env` out of version control and rotate API keys when needed.

---

## Extending the project (ideas)

* Add pagination & filtering (by genre, year, rating) to the search endpoint.
* Return source-aware answers: include citations that reference which IMDB entries the model used for each sentence.
* Add caching for repeated queries.
* Implement a React or Streamlit frontend for a more polished UI.
* Add unit tests for the ingestion scripts and API endpoints.

---

## Acknowledgements

* IMDB Top 1000 dataset
* Sentence Transformers
* Qdrant
* Google Generative AI (Gemini)

---

## License

Add your preferred license (e.g., MIT).
