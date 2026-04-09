import spacy
from fastapi import FastAPI
from pydantic import BaseModel

# 加载模型
nlp = spacy.load("en_core_web_sm")

app = FastAPI()

class TextRequest(BaseModel):
    text: str


@app.post("/parse")
def parse_text(request: TextRequest):

    doc = nlp(request.text)

    edges = []
    tokens = []

    # ----------------------------
    # tokens（🔥 新增）
    # ----------------------------
    for token in doc:
        tokens.append({
            "text": token.text,
            "lemma": token.lemma_,
            "index": token.i,
            "pos": token.pos_,
            "is_stop": token.is_stop
        })

    # ----------------------------
    # edges（你原来的）
    # ----------------------------
    for token in doc:
        if token.dep_ != "ROOT":
            edges.append({
                "headLemma": token.head.lemma_,
                "relation": token.dep_,
                "targetLemma": token.lemma_,
                "headText": token.head.text,
                "targetText": token.text,
                "headIndex": token.head.i,
                "targetIndex": token.i,
                "headPos": token.head.pos_,
                "targetPos": token.pos_,
            })

    return {
        "tokens": tokens,
        "edges": edges
    }


@app.get("/")
def root():
    return {"status": "ok"}
