from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List, Optional, Any
import os
from pinecone import Pinecone
from openai import OpenAI

# ---------------------------------------------------------
# 1. Configuration & Constants
# ---------------------------------------------------------
INDEX_NAME = "ted-rag-index"
# Using the models specified in the PDF/Assignment
EMBEDDING_MODEL = "RPRTHPB-text-embedding-3-small"
EMBEDDING_DIMENSIONS = 1536

# RAG Parameters (Hyperparameters reported in /api/stats)
TOP_K = 8
MAX_CHUNK_SIZE = 512
OVERLAP_RATIO = 0.2

# ---------------------------------------------------------
# 2. Clients Setup
# ---------------------------------------------------------
# Ensure these environment variables are set in your Vercel deployment
api_key_openai = os.environ.get("OPENAI_API_KEY")
api_key_pinecone = os.environ.get("PINECONE_API_KEY")

# Client for the specified LLM API (LLMod.ai)
client = OpenAI(
    api_key=api_key_openai,
    base_url="https://api.llmod.ai/v1" 
)

# Pinecone Client
if api_key_pinecone:
    pc = Pinecone(api_key=api_key_pinecone)
    index = pc.Index(INDEX_NAME)
else:
    index = None
    print("Warning: PINECONE_API_KEY missing.")

# ---------------------------------------------------------
# 3. FastAPI App Setup
# ---------------------------------------------------------
app = FastAPI(title="TED RAG Assistant")

# ---------------------------------------------------------
# 4. Data Models
# ---------------------------------------------------------
class QueryRequest(BaseModel):
    question: str

class ContextItem(BaseModel):
    talk_id: str
    title: str
    chunk: str   # Renamed from 'text' to 'chunk' to match PDF requirement
    score: float
    # Optional extra metadata helpful for the frontend
    url: Optional[str] = None
    speaker: Optional[str] = None
    topics: Optional[str] = None
    views: Optional[int] = None
    published_date: Optional[str] = None

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
# 5. UI (HTML/JS)
# ---------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
def read_root():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>TED Talk RAG Assistant</title>
        <style>
            body { font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif; max-width: 900px; margin: 0 auto; padding: 40px 20px; background-color: #f4f4f4; color: #333; }
            h1 { color: #e62b1e; text-align: center; margin-bottom: 10px; letter-spacing: -1px; }
            .subtitle { text-align: center; color: #666; margin-bottom: 40px; }
            .container { background: white; padding: 40px; border-radius: 8px; box-shadow: 0 10px 25px rgba(0,0,0,0.05); }
            
            .input-group { display: flex; gap: 10px; margin-bottom: 20px; }
            input[type="text"] { flex-grow: 1; padding: 15px; border: 2px solid #ddd; border-radius: 6px; font-size: 16px; transition: border-color 0.3s; }
            input[type="text"]:focus { border-color: #e62b1e; outline: none; }
            
            button { background-color: #e62b1e; color: white; border: none; padding: 15px 30px; border-radius: 6px; cursor: pointer; font-size: 16px; font-weight: bold; transition: background-color 0.3s; }
            button:hover { background-color: #c41e12; }
            button:disabled { background-color: #ccc; cursor: not-allowed; }
            
            .loader { display: none; text-align: center; margin: 20px 0; }
            .spinner { border: 4px solid #f3f3f3; border-top: 4px solid #e62b1e; border-radius: 50%; width: 30px; height: 30px; animation: spin 1s linear infinite; margin: 0 auto 10px; }
            @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }

            .result-section { display: none; margin-top: 30px; animation: fadeIn 0.5s; }
            @keyframes fadeIn { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
            
            .answer-box { background-color: #f8fff9; border-left: 5px solid #28a745; padding: 20px; border-radius: 4px; font-size: 1.1em; line-height: 1.6; color: #2c3e50; margin-bottom: 30px; box-shadow: 0 2px 5px rgba(0,0,0,0.05); }
            
            /* --- Video Section Styles --- */
            .video-section { margin-top: 25px; padding-top: 20px; border-top: 2px dashed #e1e1e1; }
            .video-header { font-weight: bold; color: #e62b1e; margin-bottom: 10px; display: flex; align-items: center; gap: 10px; }
            .video-wrapper { position: relative; padding-bottom: 56.25%; /* 16:9 */ height: 0; overflow: hidden; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.15); background: #000; }
            .video-wrapper iframe { position: absolute; top: 0; left: 0; width: 100%; height: 100%; border: 0; }

            /* Metadata Styles */
            .metadata-container { border: 1px solid #eee; border-radius: 6px; overflow: hidden; margin-top: 20px; }
            .metadata-header { background: #f9f9f9; padding: 15px; cursor: pointer; display: flex; justify-content: space-between; align-items: center; font-weight: 600; font-size: 0.9em; color: #555; }
            .metadata-header:hover { background: #f0f0f0; }
            .metadata-content { display: none; padding: 20px; background: #fff; border-top: 1px solid #eee; }
            
            .chunk-card { background: #fdfdfd; border: 1px solid #eee; padding: 15px; margin-bottom: 15px; border-radius: 6px; font-size: 0.9em; position: relative; }
            .chunk-meta { display: flex; justify-content: space-between; margin-bottom: 8px; font-size: 0.85em; color: #888; text-transform: uppercase; letter-spacing: 0.5px; }
            .chunk-title { color: #e62b1e; font-weight: bold; font-size: 1.1em; text-decoration: none; }
            .chunk-title a { color: #e62b1e; text-decoration: none; }
            .chunk-title a:hover { text-decoration: underline; }
            .chunk-info { font-size: 0.85em; color: #555; margin-bottom: 8px; background: #eee; display: inline-block; padding: 2px 6px; border-radius: 4px; }
            .chunk-text { color: #444; font-style: italic; margin-top: 5px; line-height: 1.4; }
            
            pre { white-space: pre-wrap; background: #2d2d2d; color: #ccc; padding: 15px; border-radius: 6px; font-size: 0.85em; overflow-x: auto; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>TED Talk RAG Assistant</h1>
            <p class="subtitle">Ask questions based strictly on the TED dataset context.</p>
            
            <div class="input-group">
                <input type="text" id="questionInput" placeholder="E.g., What does Ken Robinson say about creativity?" onkeypress="handleEnter(event)">
                <button onclick="sendQuery()" id="submitBtn">Ask</button>
            </div>

            <div class="loader" id="loader">
                <div class="spinner"></div>
                <p>Analyzing talks & generating answer...</p>
            </div>

            <div class="result-section" id="results">
                <h3>Answer:</h3>
                <div class="answer-box" id="answerDisplay"></div>

                <div class="metadata-container">
                    <div class="metadata-header" onclick="toggleSection('contextContent')">
                        <span>📚 Retrieved Context (Sources)</span>
                        <span>▼</span>
                    </div>
                    <div class="metadata-content" id="contextContent">
                        <div id="chunksList"></div>
                    </div>
                </div>

                <div class="metadata-container">
                    <div class="metadata-header" onclick="toggleSection('promptContent')">
                        <span>🛠️ Debug: Prompts</span>
                        <span>▼</span>
                    </div>
                    <div class="metadata-content" id="promptContent">
                        <strong>System Prompt:</strong>
                        <pre id="systemPromptDisplay"></pre>
                        <strong>User Prompt:</strong>
                        <pre id="userPromptDisplay"></pre>
                    </div>
                </div>
            </div>
        </div>

        <script>
            function handleEnter(e) { if (e.key === 'Enter') sendQuery(); }

            function toggleSection(id) {
                const el = document.getElementById(id);
                el.style.display = (el.style.display === 'block') ? 'none' : 'block';
            }

            async function sendQuery() {
                const question = document.getElementById('questionInput').value;
                if (!question) return;

                document.getElementById('submitBtn').disabled = true;
                document.getElementById('loader').style.display = 'block';
                document.getElementById('results').style.display = 'none';

                try {
                    const response = await fetch('/api/prompt', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ question: question })
                    });
                    
                    const data = await response.json();
                    
                    // 1. Display Text Answer
                    let answerHTML = data.response.replace(/\\n/g, '<br>');

                    // 2. Video Embedding Logic
                    if (data.context && data.context.length > 0 && data.context[0].url) {
                        const topMatch = data.context[0];
                        let videoUrl = topMatch.url;
                        
                        // Convert standard TED URL to Embed URL
                        if (videoUrl.includes("ted.com/talks")) {
                            videoUrl = videoUrl.replace("www.ted.com", "embed.ted.com");
                            if (!videoUrl.includes("embed.ted.com")) {
                                videoUrl = videoUrl.replace("ted.com", "embed.ted.com");
                            }
                        }

                        answerHTML += `
                            <div class="video-section">
                                <div class="video-header">
                                    <span>🎬 Watch Top Result:</span>
                                    <a href="${topMatch.url}" target="_blank" style="font-size:0.9em;">(Open in TED)</a>
                                </div>
                                <div class="video-wrapper">
                                    <iframe src="${videoUrl}" allowfullscreen scrolling="no"></iframe>
                                </div>
                            </div>
                        `;
                    }

                    document.getElementById('answerDisplay').innerHTML = answerHTML;

                    // 3. Display Metadata Chunks
                    const chunksList = document.getElementById('chunksList');
                    chunksList.innerHTML = '';
                    
                    if (!data.context || data.context.length === 0) {
                        chunksList.innerHTML = '<p>No relevant context found.</p>';
                    } else {
                        data.context.forEach(ctx => {
                            const div = document.createElement('div');
                            div.className = 'chunk-card';
                            
                            const link = ctx.url ? `<a href="${ctx.url}" target="_blank">${ctx.title}</a>` : ctx.title;
                            const speaker = ctx.speaker ? `🗣️ ${ctx.speaker}` : '';
                            const views = ctx.views ? `👁️ ${ctx.views.toLocaleString()} views` : '';
                            const date = ctx.published_date ? `📅 ${ctx.published_date}` : '';

                            div.innerHTML = `
                                <div class="chunk-meta">
                                    <span>ID: ${ctx.talk_id}</span>
                                    <span>Score: ${(ctx.score * 100).toFixed(1)}%</span>
                                </div>
                                <div class="chunk-title">${link}</div>
                                <div class="chunk-info">
                                    ${speaker} &nbsp;|&nbsp; ${views} &nbsp;|&nbsp; ${date}
                                </div>
                                <div class="chunk-text">"...${ctx.chunk}..."</div>
                            `;
                            chunksList.appendChild(div);
                        });
                    }

                    document.getElementById('systemPromptDisplay').innerText = data.Augmented_prompt.System;
                    document.getElementById('userPromptDisplay').innerText = data.Augmented_prompt.User;

                    document.getElementById('results').style.display = 'block';

                } catch (error) {
                    alert('Error: ' + error.message);
                } finally {
                    document.getElementById('submitBtn').disabled = false;
                    document.getElementById('submitBtn').innerText = 'Ask';
                    document.getElementById('loader').style.display = 'none';
                }
            }
        </script>
    </body>
    </html>
    """

# ---------------------------------------------------------
# 6. API Endpoints
# ---------------------------------------------------------

@app.get("/api/stats", response_model=StatsResponse)
def stats_endpoint():
    """
    Returns the RAG system hyperparameters.
    Strict JSON format required by assignment: chunk_size, overlap_ratio, top_k
    """
    return {
        "chunk_size": MAX_CHUNK_SIZE,
        "overlap_ratio": OVERLAP_RATIO,
        "top_k": TOP_K
    }

@app.post("/api/prompt", response_model=QueryResponse)
def prompt_endpoint(request: QueryRequest):
    if not index:
         raise HTTPException(status_code=500, detail="Server Error: Pinecone not initialized.")

    question = request.question
    
    # A. Embed Question
    try:
        res_embed = client.embeddings.create(input=question, model=EMBEDDING_MODEL)
        q_embed = res_embed.data[0].embedding
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding Error: {str(e)}")

    # B. Retrieve Context from Pinecone
    try:
        search_results = index.query(vector=q_embed, top_k=TOP_K, include_metadata=True)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pinecone Query Error: {str(e)}")

    # C. Build Context & Extract Metadata
    context_text = ""
    retrieved_chunks = []
    
    for match in search_results['matches']:
        if not match.metadata: continue
        meta = match.metadata
        
        # 1. Extract Text
        # Note: Your metadata might call it 'chunk_text', 'text', or 'chunk'
        chunk_content = str(meta.get('chunk_text', ''))
        if not chunk_content:
             chunk_content = str(meta.get('chunk', '')) # Fallback
        if not chunk_content:
             chunk_content = str(meta.get('text', '')) # Final Fallback

        # 2. Extract Metadata
        t_id = str(meta.get('talk_id', 'Unknown'))
        title = str(meta.get('title', 'Unknown'))
        speaker = str(meta.get('speaker', 'Unknown'))
        url = str(meta.get('url', ''))
        topics = str(meta.get('topics', ''))
        published_date = str(meta.get('published_date', ''))
        
        # Handle numerical view count safely
        try:
            views = int(float(meta.get('views', 0)))
        except:
            views = 0

        # 3. Build Rich Context String
        context_text += f"""
### Source Document
Title: {title}
Speaker: {speaker}
Date: {published_date}
Views: {views}
Link: {url}
Content: {chunk_content}
"""
        
        # 4. Populate Response List
        # Important: Mapping content to 'chunk' key as per PDF requirements
        retrieved_chunks.append({
            "talk_id": t_id,
            "title": title,
            "chunk": chunk_content, # Required key: 'chunk'
            "score": match.score,
            "speaker": speaker,
            "url": url,
            "topics": topics,
            "views": views,
            "published_date": published_date
        })

    # D. Construct Prompts
    # STRICTLY REQUIRED SYSTEM PROMPT from PDF Page 4
    system_prompt = """You are a TED Talk assistant that answers questions strictly and only based on the TED dataset context provided to you (metadata and transcript passages). You must not use any external knowledge, the open internet, or information that is not explicitly contained in the retrieved context. If the answer cannot be determined from the provided context, respond: "I don't know based on the provided TED data." Always explain your answer using the given context, quoting or paraphrasing the relevant transcript or metadata when helpful."""

    user_prompt = f"""Analyze the provided TED Talk context chunks to answer the user's question.

Determine the nature of the question (Fact, List, Summary, or Recommendation) and answer accordingly.

Context:
{context_text}

Question: {question}
Answer:"""

    # E. Generate Response (LLM)
    try:
        response = client.chat.completions.create(
            model="RPRTHPB-gpt-5-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=1 
        )
        content = response.choices[0].message.content
        answer = content if content else "Error: Model returned empty response."
        
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
