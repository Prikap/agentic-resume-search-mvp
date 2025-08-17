import json,re,requests,torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
import os


load_dotenv()
HF_API_TOKEN = os.getenv("HF_API_TOKEN")
if not HF_API_TOKEN:
    raise ValueError("Hugging Face API key not found in .env file")
HF_API_URL = "https://api-inference.huggingface.co/models/gpt2" 
headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
# This script searches resumes based on a job query, extracting skills and matching candidates.


def extract_skills_from_query_llm(query):
    prompt = f"Extract main skills and aliases/related technologies from this job query in a comma-separated list:\n{query}\nSkills:"
    payload = {"inputs": prompt, "options": {"use_cache": False, "wait_for_model": True}}
    try:
        response = requests.post(HF_API_URL, headers=headers, json=payload, timeout=10)
        text = response.json()[0]['generated_text']
        skills = [s.strip().lower() for s in text.split(",") if s.strip()]
        return list(set(skills))
    except:
        # fallback to basic regex if API fails
        return re.findall(r'\b(machine learning|deep learning|nlp|python|react|node\.js|mongodb|typescript|dsa|system design|sql|javascript)\b', query.lower())


def load_resumes(path="resumes.json"):
    with open(path, "r") as f:
        resumes = json.load(f)
    normalized = [{k.lower(): v for k, v in r.items()} for r in resumes]
    return normalized


embed_model = SentenceTransformer('all-MiniLM-L6-v2')
def embed_texts(texts):
    emb = embed_model.encode(texts, convert_to_tensor=True)
    return emb.cpu().numpy()


def parse_min_experience(query):
    match = re.search(r'(\d+)\+?\s*years', query.lower())
    return int(match.group(1)) if match else 0


def location_score(candidate_loc, query):
    candidate_loc = candidate_loc.lower()
    query = query.lower()
    for city in candidate_loc.split(","):
        if city.strip() in query:
            return 0.1
    return 0


MODEL_NAME = "BAAI/bge-reranker-v2-m3"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
bge_model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
bge_model.eval()
#bge-reranker-v2-m3 is a model for reranking, it expects two inputs: query and candidate summary.
#The output is a score between 0 and 1, where 1 means the candidate is the most relevant to the query.

def compute_bge_score(query, candidate_summary):
    inputs = tokenizer(query, candidate_summary, return_tensors='pt', truncation=True, padding=True, max_length=1024)
    with torch.no_grad():
        outputs = bge_model(**inputs)
    score = torch.sigmoid(outputs.logits).item()
    return score




def generate_explanation(candidate, query_skills, matched_skills, critical_missing, min_exp):
    explanation = f"Candidate meets minimum experience ({candidate['experience']} ≥ {min_exp} years). "
    explanation += f"Matched skills: {matched_skills if matched_skills else 'None'}. "
    explanation += f"Critical missing skills: {critical_missing if critical_missing else 'None'}. "
    explanation += f"Location: {candidate['location']}."
    return explanation



def search_candidates(query, resumes, top_k=5):
    min_exp = parse_min_experience(query)
    query_skills = extract_skills_from_query_llm(query)
    query_norm = " ".join(query_skills + re.findall(r'\w+', query.lower()))
    query_emb = embed_texts([query_norm])

    results = []
    for r in resumes:
        candidate_exp = r.get("years_of_experience",0)
        if candidate_exp < min_exp:
            continue

        candidate_skills = r.get("skills", [])
        skills_text = " ".join([s.lower() for s in candidate_skills])
        skills_emb = embed_texts([skills_text])
        skill_score = cosine_similarity(query_emb, skills_emb)[0][0]

        summary_emb = embed_texts([r.get("summary","")])
        summary_score = cosine_similarity(query_emb, summary_emb)[0][0]

        loc_bonus = location_score(r.get("location",""), query)
        exp_bonus = min(candidate_exp / max(min_exp,1),1.0) * 0.1

        bge_score = compute_bge_score(query, r.get("summary",""))
        

        total_score = skill_score*0.25 + summary_score*0.15 + bge_score*0.45 + loc_bonus + exp_bonus


        matched_skills = [s for s in candidate_skills if s.lower() in query_skills]
        critical_missing = [s for s in query_skills if s not in [m.lower() for m in matched_skills]]

        explanation = generate_explanation(
            candidate={"experience": candidate_exp,"location": r.get("location","N/A")},
            query_skills=query_skills,
            matched_skills=matched_skills,
            critical_missing=critical_missing,
            min_exp=min_exp
        )

        results.append({
            "candidate_name": r.get("candidate_name","N/A"),
            "score": float(round(total_score,2)),
            "matched_skills": matched_skills,
            "critical_missing_skills": critical_missing,
            "experience": candidate_exp,
            "location": r.get("location","N/A"),
            "summary": r.get("summary",""),
            "explanation": explanation
        })

    results = sorted(results, key=lambda x: x['score'], reverse=True)

    with open("top_candidates.json","w") as f:
        json.dump(results[:top_k], f, indent=2)

    return results[:top_k]

if __name__ == "__main__":
    resumes = load_resumes("resumes.json")
    query = input("Enter your query: ")
    top_candidates = search_candidates(query, resumes)
    print(json.dumps(top_candidates, indent=2))