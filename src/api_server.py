from fastapi import FastAPI # type: ignore
from pydantic import BaseModel # type: ignore
from src.model import RAGModel

app = FastAPI()
model = RAGModel()

class QuestionRequest(BaseModel):
    question: str

@app.post("/api/answer")
async def get_answer(request: QuestionRequest):
    answer = model.generate_answer(request.question)[0]
    return {"question": request.question, "answer": answer}

if __name__ == "__main__":
    import uvicorn # type: ignore
    uvicorn.run(app, host="0.0.0.0", port=8000)
