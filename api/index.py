from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import os
from pinecone import Pinecone
from openai import OpenAI

# --- Configuration ---
INDEX_NAME = "ted-rag-index"
EMBEDDING_MODEL = "RPRTHPB-text-embedding-3-small"
EMBEDDING_DIMENSIONS = 1536
TOP_K = 5
MAX_CHUNK_SIZE = 2048
OVERLAP_RATIO = 0.2

# --- Clients Setup ---
# Keys will be loaded from Vercel Environment Variables
api_key_openai = os.environ.get("OPENAI_API_KEY")
api_key_pinecone = os.environ.get("PINECONE_API_KEY")

client = OpenAI(
    api_key=api_key_openai,
    base_url="https://api.llmod.ai/v1"
)

# Initialize Pinecone only if key exists (to avoid build errors)
if api_key_pinecone:
    pc = Pinecone(api_key=api_key_pinecone)
    index = pc.Index(INDEX_NAME)

# --- FastAPI App ---
app = FastAPI()

# --- Data Models (JSON Schema) ---
class QueryRequest(BaseModel):
    question: str

class ContextItem(BaseModel):
    talk_id: str
    title: str
    chunk: str
    score: float

class AugmentedPrompt(BaseModel):
    System: str
    User: str

class QueryResponse(BaseModel):
    response: str
    context: List[ContextItem]
    Augmented_prompt: AugmentedPrompt

class StatsResponse(BaseModel):
    chunk_size: int
    overlap_ratio: float
    top_k: int

# --- Core Logic ---
@app.get("/api/stats", response_model=StatsResponse)
def stats_endpoint():
    return {
        "chunk_size": MAX_CHUNK_SIZE,
        "overlap_ratio": OVERLAP_RATIO,
        "top_k": TOP_K
    }

@app.post("/api/prompt", response_model=QueryResponse)
def prompt_endpoint(request: QueryRequest):
    question = request.question
    
    # 1. Embed
    try:
        res_embed = client.embeddings.create(input=question, model=EMBEDDING_MODEL)
        q_embed = res_embed.data[0].embedding
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding Error: {str(e)}")

    # 2. Retrieve
    try:
        search_results = index.query(vector=q_embed, top_k=TOP_K, include_metadata=True)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pinecone Error: {str(e)}")

    # 3. Build Context
    context_text = ""
    retrieved_chunks = []
    
    for match in search_results['matches']:
        if not match.metadata: continue
        meta = match.metadata
        
        # Safe extraction
        t_id = str(meta.get('talk_id', ''))
        title = str(meta.get('title', ''))
        chunk = str(meta.get('chunk', ''))
        
        context_text += f"\n### Source Document\nTitle: {title}\nContent: {chunk}\n"
        
        retrieved_chunks.append({
            "talk_id": t_id,
            "title": title,
            "chunk": chunk,
            "score": match.score
        })

    # 4. Prompts (Strict System, Smart User)
    system_prompt = """You are a TED Talk assistant that answers questions strictly and only based on the TED dataset context provided to you (metadata and transcript passages).
You must not use any external knowledge, the open internet, or information that is not explicitly contained in the retrieved context.
If the answer cannot be determined from the provided context, respond: "I don't know based on the provided TED data."
Always explain your answer using the given context, quoting or paraphrasing the relevant transcript or metadata when helpful.
You may add additional clarifications (e.g., response style), but you must keep the above constraints."""

    user_prompt = f"""Please analyze the retrieved TED Talk context chunks below to answer the user's question.
Trust the provided text; the answer is likely contained strictly within these excerpts.

Determine which of the following 4 categories the question falls into and format your answer accordingly:
1. **Precise Fact Retrieval**: Provide the specific Entity, Title, and Speaker.
2. **Multi-Result Topic Listing**: Return a list of exactly 3 talk titles (or fewer if less found).
3. **Key Idea Summary Extraction**: Provide Title and concise summary.
4. **Recommendation**: Choose ONE talk and justify.

Context:
{context_text}

Question: {question}
Answer:"""

    # 5. Generate
    try:
        response = client.chat.completions.create(
            model="RPRTHPB-gpt-5-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=1 
        )
        answer = response.choices[0].message.content
    except Exception as e:
        answer = f"Error generating response: {str(e)}"

    return {
        "response": answer,
        "context": retrieved_chunks,
        "Augmented_prompt": {
            "System": system_prompt,
            "User": user_prompt
        }
    }