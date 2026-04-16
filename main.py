import spacy
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import JSONResponse

# =========================
# LOAD MODEL（🔥关键）
# =========================
try:
    nlp = spacy.load("en_core_web_sm")
except:
    # 防止 Render 没装模型
    import subprocess
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")

# =========================
# APP
# =========================
app = FastAPI()


# =========================
# REQUEST MODEL
# =========================
class TextRequest(BaseModel):
    text: str


# =========================
# MAIN PARSE API
# =========================
@app.post("/parse")
def parse_text(request: TextRequest):
    try:
        text = request.text.strip()

        if not text:
            return JSONResponse(
                content={"error": "empty text"},
                status_code=400
            )

        doc = nlp(text)

        tokens = []
        edges = []

        # =========================
        # TOKENS（🔥 matcher核心）
        # =========================
        for token in doc:
            tokens.append({
                "text": token.text,
                "lemma": token.lemma_,
                "index": token.i,
                "pos": token.pos_,
                "tag": token.tag_,              # 🔥 for possessive
                "dep": token.dep_,              # 🔥 for relation
                "head": token.head.i,           # 🔥 for relation
                "is_stop": token.is_stop
            })

        # =========================
        # EDGES（可选 debug）
        # =========================
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

    except Exception as e:
        return JSONResponse(
            content={"error": str(e)},
            status_code=500
        )


# =========================
# DEBUG ENDPOINT（🔥强烈推荐）
# =========================
@app.post("/debug")
def debug_text(request: TextRequest):
    doc = nlp(request.text)

    return [
        {
            "text": t.text,
            "lemma": t.lemma_,
            "pos": t.pos_,
            "tag": t.tag_,
            "dep": t.dep_,
            "head": t.head.text
        }
        for t in doc
    ]


# =========================
# HEALTH CHECK
# =========================
@app.get("/")
def root():
    return {"status": "ok"}
