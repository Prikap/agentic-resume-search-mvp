# Agentic Resume Search MVP

## Overview
This project implements an **agentic resume search engine** that ranks candidate resumes based on a natural language query, considering skills, experience, and location.

## Features
- Dynamic skill & alias extraction using Hugging Face API  
- Semantic similarity scoring with Sentence Transformers  
- BGE reranker for improved contextual relevance  
- Skills, Minimum experience and location filtering  
- JSON output with explanations for candidate selection  

## Installation
git clone <repo_url>
cd agentic_resume_search_mvp
pip install -r requirements.txt

Add your Hugging Face API key to .env
HF_API_TOKEN=your_api_key_here

## Usage
python agentic_resume_search_full_env.py
Enter a natural language query when prompted.

Results are ranked and saved in top_candidates.json.


# agentic-resume-search-mvp
