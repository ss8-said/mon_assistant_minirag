from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from rag import build_rag_chain

app = FastAPI()
qa_chain = build_rag_chain()

# Autoriser React depuis Vercel
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Remplace * par ton domaine Vercel si tu veux sécuriser
    allow_methods=["*"],
    allow_headers=["*"],
)

class Query(BaseModel):
    text: str

@app.post("/chat")
async def chat(query: Query):
    response = qa_chain.run(query.text)
    return {"response": response}
