# 🧠 TED Talk RAG Assistant

> **A high-performance RAG (Retrieval-Augmented Generation) engine for thousands of TED Talks.**
> Answers questions, retrieves precise video segments, and provides context-aware insights using Vector Search and LLMs.

[![Deployment](https://img.shields.io/badge/Vercel-Live_Demo-black?logo=vercel&style=for-the-badge)](https://ted-cx9q8ym1h-yoniafek1s-projects.vercel.app/)
[![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python&style=for-the-badge)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-High_Performance-009688?logo=fastapi&style=for-the-badge)](https://fastapi.tiangolo.com/)
[![Pinecone](https://img.shields.io/badge/Vector_DB-Pinecone-red?style=for-the-badge)](https://www.pinecone.io/)
[![OpenAI](https://img.shields.io/badge/LLM-GPT--5--mini-green?logo=openai&style=for-the-badge)](https://openai.com/)
---
Here is a short explanation in English for your `README.md`:

-----

### 📡 API Endpoints

1.  **`POST /api/prompt`**

      * **Purpose:** The main RAG endpoint. It processes your question, retrieves relevant context from Pinecone, and generates an answer.
      * **How to test:** Since this is a POST request, use the Swagger UI at [/docs](https://www.google.com/search?q=https://ted-rag.vercel.app/docs). Locate `/api/prompt`, click **Try it out**, and send a JSON body with your `question`.

2.  **`GET /api/stats`**

      * **Purpose:** Returns the current system hyperparameters (chunk size, overlap ratio, and top-k).
      * **Link:** [https://ted-rag.vercel.app/api/stats](https://www.google.com/search?q=https://ted-rag.vercel.app/api/stats)
    
---

## 🚀 Live Demo
Experience the assistant in action:
### 👉 [Click here to visit the Live App](https://ted-cx9q8ym1h-yoniafek1s-projects.vercel.app/)

---

<p align="center">
  <img width="600" src="https://github.com/user-attachments/assets/e9fec509-76eb-4ead-9532-f1f1cc898cd9" alt="TED Koala">
</p>

---

## 📖 About The Project

This project implements a robust **Retrieval-Augmented Generation (RAG)** system designed to navigate a massive dataset of TED Talk transcripts. unlike simple keyword search, this assistant understands semantic intent, allowing users to ask complex questions and receive grounded, evidence-based answers.

The system processes thousands of hours of spoken content, indexing them into a vector database to allow for millisecond-latency retrieval of relevant contexts.

### ✨ Key Features
* **Semantic Understanding:** Retrieves relevant talks even if the exact keywords aren't used.
* **Rich Metadata Integration:** Answers include the **Speaker**, **View Count**, **Publish Date**, and a direct **URL** to the video.
* **Smart Chunking:** Utilizes a sliding window approach (512 tokens with 20% overlap) to preserve narrative flow and context.
* **Hallucination Guardrails:** The LLM is strictly prompted to answer *only* based on retrieved evidence, ensuring factual accuracy.
* **Clean UI:** A responsive, user-friendly interface built with Vanilla JS and HTML5.

---

## 🛠️ Architecture & Workflow

The system is built on a modern ETL (Extract, Transform, Load) and Retrieval pipeline:

```mermaid
graph TD
    A[Raw TED CSV Dataset] -->|Clean & Process| B(Text Chunking)
    B --> C{Embedding Model}
    C -->|text-embedding-3-small| D[(Pinecone Vector DB)]
    
    User[User Question] -->|Embed Query| E(Vector Search)
    E -->|Top-k Similarity| D
    D -->|Retrieve Context + Metadata| F[LLM Construction]
    F -->|System Prompt + Context| G[GPT-5-mini]
    G --> Output[Final Answer]
````

### 🧰 Tech Stack

  * **Backend:** Python, FastAPI, Uvicorn.
  * **AI Engine:** OpenAI API (Embeddings + ChatCompletion).
  * **Vector Database:** Pinecone (Serverless).
  * **Data Processing:** Pandas, Tiktoken.
  * **Frontend:** HTML5, CSS3, JavaScript (Fetch API).
  * **Deployment:** Vercel.

-----

## 📂 The Dataset

The system is trained on a comprehensive dataset of TED Talks. The data pipeline extracts and indexes the following fields for every talk:

| Field | Description | Usage |
| :--- | :--- | :--- |
| **Transcript** | Full text of the speech | Source for vector embeddings and semantic search. |
| **Title** | Talk title | Identification and citation. |
| **Speaker** | Primary speaker name | Context and attribution. |
| **Views** | View count | Used as a metric for popularity/recommendations. |
| **URL** | Link to TED.com | Direct access for the user. |
| **Topics** | Tags/Categories | Context enrichment. |

-----

## 💻 Local Installation

Follow these steps to run the project locally on your machine.

### 1\. Clone the Repository

```bash
git clone [https://github.com/YoniAfek1/ted-rag-assistant.git](https://github.com/YoniAfek1/ted-rag-assistant.git)
cd ted-rag-assistant
```

### 2\. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3\. Environment Configuration

Create a `.env` file in the root directory or export the variables directly:

```bash
export OPENAI_API_KEY="your_openai_key_here"
export PINECONE_API_KEY="your_pinecone_key_here"
```

### 4\. Data Ingestion (ETL)

*Note: Run this once to populate your Pinecone index.*

```bash
python etl_pipeline.py
```

This script will:

1.  Load the CSV dataset.
2.  Chunk the transcripts.
3.  Generate embeddings.
4.  Upsert vectors and metadata to Pinecone.

### 5\. Run the Server

```bash
uvicorn main:app --reload
```

The application will be available at: `http://127.0.0.1:8000`

-----

## 🔍 API Reference

The backend exposes the following REST endpoints:

  * `GET /` - Serves the frontend UI.
  * `GET /api/stats` - Returns current RAG configuration (chunk size, overlap, etc.).
  * `POST /api/prompt` - The main RAG endpoint.
      * **Body:** `{"question": "Your question here"}`
      * **Response:** JSON containing the LLM answer, source chunks, metadata, and debug info.
