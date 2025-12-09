from fastapi import FastAPI

app = FastAPI()

# 1. בדיקת דף הבית (מוודא שהשרת חי)
@app.get("/")
def read_root():
    return {
        "status": "Alive",
        "message": "Vercel deployment is working perfectly!",
        "next_step": "Now add your API keys and uncomment the logic."
    }

# 2. בדיקת נתיב API (מוודא שה-vercel.json עובד ומפנה נכון)
@app.get("/api/stats")
def read_stats():
    return {
        "test_mode": True,
        "chunk_size": 2048,
        "top_k": 5,
        "note": "This is mock data. No Pinecone connection yet."
    }

# 3. בדיקת POST (מוודא שאפשר לשלוח מידע)
@app.post("/api/prompt")
def read_prompt(data: dict):
    return {
        "received_question": data.get("question"),
        "mock_response": "I am a dummy response because API keys are not active yet.",
        "context": []
    }
