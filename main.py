from fastapi import FastAPI
from pydantic import BaseModel
import spacy

# ✅ 加载模型（只会加载一次）
nlp = spacy.load("en_core_web_sm")

# ✅ 初始化 API
app = FastAPI()

# ✅ 定义请求格式
class TextRequest(BaseModel):
    text: str

# ✅ 核心接口
@app.post("/parse")
def parse_text(request: TextRequest):
    doc = nlp(request.text)

    edges = []

    for token in doc:
        # 过滤 ROOT
        if token.dep_ != "ROOT":
            edges.append({
                "headLemma": token.head.lemma_,
                "relation": token.dep_,
                "targetLemma": token.lemma_,
                "headText": token.head.text,
                "targetText": token.text,
                "headIndex": token.head.i,
                "targetIndex": token.i
            })

    return edges

# ✅ 健康检查（部署用）
@app.get("/")
def root():
    return {"status": "ok"}
