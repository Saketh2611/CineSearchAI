from fastapi import FastAPI, Request
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
import google.generativeai as genai
from fastapi.templating import Jinja2Templates
from typing import List, Dict, Any
import os
from dotenv import load_dotenv

# --- Load environment variables ---
load_dotenv()

# --- Initialize FastAPI ---
app = FastAPI()

# --- Load templates (for frontend rendering if needed) ---
templates = Jinja2Templates(directory="templates")

# --- Initialize the embedding model once ---
model = SentenceTransformer("All-MiniLM-L6-v2")

# --- Initialize clients using environment variables ---
qdrant_client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY")
)

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# --- Pydantic model for request body ---
class SearchQuery(BaseModel):
    query: str

# --- Root endpoint (for homepage) ---
@app.get("/")
async def read_root(request: Request):
    """Serves the main HTML page."""
    return templates.TemplateResponse("index.html", {"request": request})

# --- Search + RAG endpoint ---
@app.post("/search/")
async def search(search_query: SearchQuery):
    try:
        # 1. Convert user query into embedding vector
        query_vector = model.encode(search_query.query).tolist()

        # 2. Perform semantic search in Qdrant
        search_result = qdrant_client.search(
            collection_name="IMDB",
            query_vector=query_vector,
            limit=5
        )

        # 3. Handle empty results
        if not search_result:
            return {"results": [], "summary": "No results found."}

        # 4. Extract payloads from search results
        results = [
            {k: hit.payload.get(k) for k in [
                "Title", "Description", "Genre", "Votes", "Rate",
                "Year", "Director", "Stars", "Metascore", "Duration", "Gross"
            ]}
            for hit in search_result
        ]

        # 5. Prepare textual context for RAG prompt
        context = "\n\n".join(
            f"Title: {r['Title']}\nDescription: {r['Description']}\nGenre: {r['Genre']}\nVotes: {r['Votes']}\nRate: {r['Rate']}\nYear: {r['Year']}\nDirector: {r['Director']}\nStars: {r['Stars']}\nMetascore: {r['Metascore']}\nDuration: {r['Duration']}\nGross: {r['Gross']}"
            for r in results if r.get("Title") and r.get("Description") and r.get("Genre") and r.get("Votes") and r.get("Rate") and r.get("Year") and r.get("Director") and r.get("Stars") and r.get("Metascore") and r.get("Duration") and r.get("Gross")
        )

        # 6. Create prompt for Gemini
        prompt = (
            f"You are a movie expert who provides a summary for a user. "
            f"Based on their query and the search results below, explain why these movies are a good match.\n\n"
            f"User Query: \"{search_query.query}\"\n\n"
            f"Search Results:\n{context}\n\nYour summary:"
            f"Give clean text dont include any bullet points or numbers and stars."
        )

        # 7. Generate AI response using Gemini
        generation_model = genai.GenerativeModel("gemini-2.5-flash")
        summary_response = generation_model.generate_content(prompt)

        # 8. Return structured response
        return {"results": results, "summary": summary_response.text}

    except Exception as e:
        return {"error": str(e)}
