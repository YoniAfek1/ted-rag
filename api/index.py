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
EMBEDDING_MODEL = "text-embedding-3-small" # וודא שזה תואם את מה שהשתמשת ב-Ingestion
EMBEDDING_DIMENSIONS = 1536
TOP_K = 5
MAX_CHUNK_SIZE = 512 # הותאם ל-Ingestion
OVERLAP_RATIO = 0.2

# ---------------------------------------------------------
# 2. Clients Setup
# ---------------------------------------------------------
api_key_openai = os.environ.get("OPENAI_API_KEY")
api_key_pinecone = os.environ.get("PINECONE_API_KEY")

client = OpenAI(
    api_key=api_key_openai,
    # base_url="https://api.llmod.ai/v1" # הסר הערה אם צריך
)

if api_key_pinecone:
    pc = Pinecone(api_key=api_key_pinecone)
    index = pc.Index(INDEX_NAME)
else:
    index = None

# ---------------------------------------------------------
# 3. FastAPI App Setup
# ---------------------------------------------------------
app = FastAPI(title="TED RAG Assistant")

# ---------------------------------------------------------
# 4. Data Models (Updated with new fields)
# ---------------------------------------------------------
class QueryRequest(BaseModel):
    question: str

# עדכון המבנה כדי לכלול את כל המטא-דאטה החדש
class ContextItem(BaseModel):
    talk_id: str
    title: str
    text: str          # היה chunk
    score: float
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
# 5. UI (HTML/JS) - נשאר זהה אך ניתן להציג בו מידע נוסף אם תרצה
# ---------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
def read_root():
    # ... (אותו HTML כמו קודם, אין צורך לשנות אלא אם תרצה להציג את ה-URL)
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>TED Talk RAG Assistant</title>
        <style>
            body { font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 40px 20px; background-color: #f4f4f4; color: #333; }
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
            .metadata-container { border: 1px solid #eee; border-radius: 6px; overflow: hidden; margin-top: 20px; }
            .metadata-header { background: #f9f9f9; padding: 15px; cursor: pointer; display: flex; justify-content: space-between; align-items: center; font-weight: 600; font-size: 0.9em; color: #555; }
            .metadata-header:hover { background: #f0f0f0; }
            .metadata-content { display: none; padding: 20px; background: #fff; border-top: 1px solid #eee; }
            .chunk-card { background: #fdfdfd; border: 1px solid #eee; padding: 15px; margin-bottom: 15px; border-radius: 6px; font-size: 0.9em; }
            .chunk-meta { display: flex; justify-content: space-between; margin-bottom: 8px; font-size: 0.85em; color: #888; text-transform: uppercase; letter-spacing: 0.5px; }
            .chunk-title { color: #e62b1e; font-weight: bold; font-size: 1.1em; }
            .chunk-details { font-size: 0.8em; color: #666; margin-bottom: 5px; } 
            .chunk-text { color: #444; font-style: italic; margin-top: 5px;}
            pre { white-space: pre-wrap; background: #2d2d2d; color: #ccc; padding: 15px; border-radius: 6px; font-size: 0.85em; overflow-x: auto; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>TED Talk RAG Assistant</h1>
            <p class="subtitle">Ask questions based strictly on the TED dataset context.</p>
            <div class="input-group">
                <input type="text" id="questionInput" placeholder="E.g., What specific actions does Al Gore suggest?" onkeypress="handleEnter(event)">
                <button onclick="sendQuery()" id="submitBtn">Ask</button>
            </div>
            <div class="loader" id="loader"><div class="spinner"></div><p>Thinking...</p></div>
            <div class="result-section" id="results">
                <h3>Answer:</h3>
                <div class="answer-box" id="answerDisplay"></div>
                <div class="metadata-container">
                    <div class="metadata-header" onclick="toggleSection('contextContent')"><span>📚 Retrieved Context</span><span>▼</span></div>
                    <div class="metadata-content" id="contextContent"><div id="chunksList"></div></div>
                </div>
                <div class="metadata-container">
                    <div class="metadata-header" onclick="toggleSection('promptContent')"><span>🛠️ Debug Info</span><span>▼</span></div>
                    <div class="metadata-content" id="promptContent">
                        <strong>System Prompt:</strong><pre id="systemPromptDisplay"></pre>
                        <strong>User Prompt:</strong><pre id="userPromptDisplay"></pre>
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
                    document.getElementById('answerDisplay').innerHTML = data.response.replace(/\\n/g, '<br>');
                    
                    const chunksList = document.getElementById('chunksList');
                    chunksList.innerHTML = '';
                    data.context.forEach(ctx => {
                        const div = document.createElement('div');
                        div.className = 'chunk-card';
                        // הוספתי תצוגה של הדובר וה-URL ב-UI
                        div.innerHTML = `
                            <div class="chunk-meta">
                                <span>ID: ${ctx.talk_id}</span>
                                <span>Score: ${(ctx.score * 100).toFixed(1)}%</span>
                            </div>
                            <div class="chunk-title"><a href="${ctx.url}" target="_blank">${ctx.title}</a></div>
                            <div class="chunk-details">Speaker: ${ctx.speaker} | Views: ${ctx.views}</div>
                            <div class="chunk-text">"...${ctx.text}..."</div>
                        `;
                        chunksList.appendChild(div);
                    });
                    document.getElementById('systemPromptDisplay').innerText = data.Augmented_prompt.System;
                    document.getElementById('userPromptDisplay').innerText = data.Augmented_prompt.User;
                    document.getElementById('results').style.display = 'block';
                } catch (error) {
                    alert('Error: ' + error.message);
                } finally {
                    document.getElementById('submitBtn').disabled = false;
                    document.getElementById('loader').style.display = 'none';
                }
            }
        </script>
    </body>
    </html>
    """

@app.get("/api/stats", response_model=StatsResponse)
def stats_endpoint():
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
    
    # A. Embed
    try:
        res_embed = client.embeddings.create(input=question, model=EMBEDDING_MODEL)
        q_embed = res_embed.data[0].embedding
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding Error: {str(e)}")

    # B. Retrieve
    try:
        search_results = index.query(vector=q_embed, top_k=TOP_K, include_metadata=True)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pinecone Query Error: {str(e)}")

    # C. Build Context (עודכן לחילוץ כל השדות)
    context_text = ""
    retrieved_chunks = []
    
    for match in search_results['matches']:
        if not match.metadata: continue
        meta = match.metadata
        
        # חילוץ בטוח של כל השדות החדשים
        # שים לב: המפתח לטקסט הוא עכשיו 'chunk_text' בהתאם לסקריפט ה-Ingestion
        t_id = str(meta.get('talk_id', 'Unknown'))
        title = str(meta.get('title', 'Unknown'))
        chunk_content = str(meta.get('chunk_text', '')) # תיקון קריטי!
        
        speaker = str(meta.get('speaker', 'Unknown'))
        url = str(meta.get('url', ''))
        topics = str(meta.get('topics', ''))
        views = meta.get('views', 0)
        published_date = str(meta.get('published_date', ''))
        
        # בניית מחרוזת קונטקסט עשירה עבור ה-LLM
        context_text += f"""
### Source Document
Title: {title}
Speaker: {speaker}
Topics: {topics}
Date: {published_date}
Views: {views}
Content: {chunk_content}
"""
        
        retrieved_chunks.append({
            "talk_id": t_id,
            "title": title,
            "text": chunk_content,
            "score": match.score,
            "speaker": speaker,
            "url": url,
            "topics": topics,
            "views": int(views) if views else 0,
            "published_date": published_date
        })

    # D. Prompts
    system_prompt = """You are a TED Talk assistant that answers questions strictly and only based on the TED dataset context provided to you.
Use the metadata (Speaker, Date, Topics, Views) to provide rich answers.
If the user asks for a recommendation, consider the 'Views' count as a metric for popularity.
If the answer cannot be determined from the provided context, respond: "I don't know based on the provided TED data."
"""

    user_prompt = f"""Analyze the retrieved TED Talk context chunks below to answer the user's question.

Context:
{context_text}

Question: {question}
Answer:"""

    # E. Generate
    try:
        response = client.chat.completions.create(
            model="RPRTHPB-gpt-5-mini", #
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=1
        )
        content = response.choices[0].message.content
        answer = content if content is not None else "Error: Empty response."
        
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
