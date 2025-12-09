from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import os
from pinecone import Pinecone
from openai import OpenAI

# ---------------------------------------------------------
# 1. Configuration & Constants
# ---------------------------------------------------------
INDEX_NAME = "ted-rag-index"
EMBEDDING_MODEL = "RPRTHPB-text-embedding-3-small"
EMBEDDING_DIMENSIONS = 1536
TOP_K = 5
MAX_CHUNK_SIZE = 2048
OVERLAP_RATIO = 0.2

# ---------------------------------------------------------
# 2. Clients Setup
# ---------------------------------------------------------
# Keys will be loaded from Vercel Environment Variables
api_key_openai = os.environ.get("OPENAI_API_KEY")
api_key_pinecone = os.environ.get("PINECONE_API_KEY")

# Initialize OpenAI Client (LLMod.ai compatible)
client = OpenAI(
    api_key=api_key_openai,
    base_url="https://api.llmod.ai/v1"
)

# Initialize Pinecone only if key exists (to avoid build errors if vars are missing)
if api_key_pinecone:
    pc = Pinecone(api_key=api_key_pinecone)
    index = pc.Index(INDEX_NAME)
else:
    print("Warning: PINECONE_API_KEY not found. Index not initialized.")
    index = None

# ---------------------------------------------------------
# 3. FastAPI App Setup
# ---------------------------------------------------------
app = FastAPI(title="TED RAG Assistant")

# ---------------------------------------------------------
# 4. Data Models (JSON Schema)
# ---------------------------------------------------------
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

# ---------------------------------------------------------
# 5. API Endpoints
# ---------------------------------------------------------

# --- Health Check (Root) ---
@app.get("/")
def read_root():
    return {
        "status": "Online",
        "message": "TED RAG API is running. Endpoints: POST /api/prompt, GET /api/stats"
    }

# --- Stats Endpoint ---
@app.get("/api/stats", response_model=StatsResponse)
def stats_endpoint():
    return {
        "chunk_size": MAX_CHUNK_SIZE,
        "overlap_ratio": OVERLAP_RATIO,
        "top_k": TOP_K
    }

# --- Prompt Endpoint (Main Logic) ---
@app.post("/api/prompt", response_model=QueryResponse)
def prompt_endpoint(request: QueryRequest):
    # Ensure Pinecone is initialized
    if not index:
         raise HTTPException(status_code=500, detail="Server Error: Pinecone not initialized (Check API Key).")

    question = request.question
    
    # A. Embed Question
    try:
        res_embed = client.embeddings.create(input=question, model=EMBEDDING_MODEL)
        q_embed = res_embed.data[0].embedding
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding Error: {str(e)}")

    # B. Retrieve Context
    try:
        search_results = index.query(vector=q_embed, top_k=TOP_K, include_metadata=True)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pinecone Query Error: {str(e)}")

    # C. Build Context String & List
    context_text = ""
    retrieved_chunks = []
    
    for match in search_results['matches']:
        if not match.metadata: continue
        meta = match.metadata
        
        # Safe extraction of metadata fields
        t_id = str(meta.get('talk_id', 'Unknown'))
        title = str(meta.get('title', 'Unknown'))
        chunk = str(meta.get('chunk', ''))
        
        # Format for the LLM
        context_text += f"\n### Source Document\nTitle: {title}\nContent: {chunk}\n"
        
        # Add to response list
        retrieved_chunks.append({
            "talk_id": t_id,
            "title": title,
            "chunk": chunk,
            "score": match.score
        })

    # D. Define Prompts
    
    # 1. System Prompt (Strict - Must match assignment requirements exactly)
    system_prompt = """You are a TED Talk assistant that answers questions strictly and only based on the TED dataset context provided to you (metadata and transcript passages).
You must not use any external knowledge, the open internet, or information that is not explicitly contained in the retrieved context.
If the answer cannot be determined from the provided context, respond: "I don't know based on the provided TED data."
Always explain your answer using the given context, quoting or paraphrasing the relevant transcript or metadata when helpful.
You may add additional clarifications (e.g., response style), but you must keep the above constraints."""

    # 2. User Prompt (Smart - Handles logic and confidence injection)
    user_prompt = f"""Please analyze the retrieved TED Talk context chunks below to answer the user's question.
Trust the provided text; the answer is likely contained strictly within these excerpts.

Determine which of the following 4 categories the question falls into and format your answer accordingly:

1. **Precise Fact Retrieval**: If the user asks for a specific detail (e.g., "Find a talk about X"), provide the specific Entity, Title, and Speaker.
2. **Multi-Result Topic Listing**: If the user asks for a list (e.g., "Which talks focus on X?"), return a list of exactly 3 talk titles (or fewer if less are found). Do not list chunks, only distinct titles.
3. **Key Idea Summary Extraction**: If the user asks for a summary of a specific theme, provide the Title and a concise summary of the main idea based on the evidence.
4. **Recommendation**: If the user asks for a recommendation (e.g., "I'm looking for a talk about..."), choose ONE relevant talk and provide a justification grounded in the data.

Context:
{context_text}

Question: {question}
Answer:"""

    # E. Generate Response via LLM
    try:
        response = client.chat.completions.create(
            model="RPRTHPB-gpt-5-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=1 # Mandatory for this model
        )
        answer = response.choices[0].message.content
    except Exception as e:
        answer = f"Error generating response: {str(e)}"

    # F. Return Final Response
    return {
        "response": answer,
        "context": retrieved_chunks,
        "Augmented_prompt": {
            "System": system_prompt,
            "User": user_prompt
        }
    }
