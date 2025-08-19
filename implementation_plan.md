# Implementation Plan (Outline)

# 1. Suggested Technologies
- Python for backend and data processing
- JSON resume dataset
- `SentenceTransformers` for embeddings (`all-MiniLM-L6-v2`)
- Hugging Face `BAAI/bge-reranker-v2-m3` for contextual relevance
- `scikit-learn` for cosine similarity
- `.env` for API key management (`python-dotenv`)

# 2. How Results Will Be Ranked
- Extract critical skills and context from the query
- Compute embeddings for query and resumes
- Calculate cosine similarity
- Use BGE reranker for contextual relevance
- Aggregate scores with weights:
  - Skill match (highest)
  - Semantic similarity
  - Reranker score
  - Location and experience bonus
- Apply minimum experience filter

# 3. How Explanations Will Be Generated
- Matched skills vs missing skills
- Experience and location
- Similarity / reranker score
- Output in JSON for each top candidate
