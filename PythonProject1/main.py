from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import pandas as pd

# Import your recommender logic
from recommender import load_data, get_recommendations, model

# Initialize FastAPI app
app = FastAPI()

# Add CORS middleware to allow all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic model for request body validation
class RecommendationRequest(BaseModel):
    query: str

# Load data and pre-compute embeddings once at startup
@app.on_event("startup")
async def startup_event():
    global movies_df, movie_embeddings
    print("Loading movie data and generating embeddings...")
    movies_df = load_data()
    features = movies_df['combined_features'].tolist()
    movie_embeddings = model.encode(features, convert_to_tensor=False)
    print("Embeddings generated successfully.")

@app.post("/recommend")
def recommend_movies(request: RecommendationRequest):
    """Endpoint to get movie recommendations based on a query."""
    user_query = request.query
    recommendations = get_recommendations(user_query, movie_df=movies_df, movie_embeddings=movie_embeddings)
    return {"recommendations": recommendations}

if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8000)
