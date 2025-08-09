Scalable Context-Aware Movie Recommendation Engine
This project is a movie recommendation system that uses modern machine learning and MLOps principles to provide personalized movie suggestions. It features a robust FastAPI backend, a responsive frontend, and is fully containerized using Docker for easy deployment and scalability.

Key Features
Context-Aware Recommendations: Utilizes state-of-the-art NLP with sentence-transformers to generate contextual embeddings for movies and user queries.

Vector Search: Employs cosine similarity to efficiently find and rank the most relevant movies based on a given text query.

FastAPI Backend: Built with a high-performance Python web framework to serve recommendations via a RESTful API.

Docker Containerization: The entire application is containerized, ensuring a consistent and reproducible environment for development and deployment.

Production-Ready Architecture: The design separates the frontend from the backend, demonstrating proficiency in building scalable, real-world data products.

Technologies Used
Python: The core language for the backend and machine learning logic.

FastAPI: For building the high-speed and asynchronous API.

sentence-transformers: A powerful library for generating sentence and text embeddings.

pandas & scikit-learn: For data preprocessing and calculating cosine similarity.

Docker: For containerizing the application, making it platform-independent and easy to deploy.

HTML/CSS/JavaScript: For the simple, interactive web-based frontend.

How to Run Locally
Prerequisites
Docker Desktop installed and running.

Steps
Clone the repository:

git clone [your-repo-url]
cd [your-repo-name]

Build the Docker image:
This command builds the container image using the Dockerfile. The first build may take a while as it downloads the necessary libraries, but subsequent builds will be much faster due to caching.

docker build -t movie-recommender .

Run the Docker container:
This command starts the application in the background and exposes it on port 8000.

docker run -d -p 8000:8000 movie-recommender

Access the application:
Open your web browser and navigate to the index.html file located in the project's root directory to use the frontend.
