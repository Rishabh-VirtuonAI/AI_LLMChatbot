from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from random import choice
from app.db import get_answers_from_db
from app.gptj_model import gptj

app = FastAPI()

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic model for question input
class Question(BaseModel):
    question: str

# Function to generate a new answer using GPT
def generate_gpt_answer(question: str):
    generated_answer = gptj.generate_answer(question)
    return generated_answer

# Dictionary to track the history of questions asked by users
asked_questions_history = {}

@app.get("/")
async def root():
    return {"Message": "Welcome to the AI Chatbot"}

@app.post("/answer")
async def get_answer(question_data: Question):
    """
    Endpoint to get an answer to the provided question.
    Expects a JSON payload with a "question" field.
    Example: { "question": "What is AI?" }
    """
    question = question_data.question.strip()

    if not question:
        raise HTTPException(status_code=400, detail="Question parameter is required")

    # Normalize question for database lookup
    normalized_question = question.lower()
    answers = get_answers_from_db(normalized_question)

    if answers:
        sampled_answers = [answer[0] for answer in answers]
        selected_answer = choice(sampled_answers)
        asked_questions_history[question] = 'answered_from_db'
        return {"answer": selected_answer}
    else:
        # Generate answer using GPT if not found in DB
        generated_answer = generate_gpt_answer(question)
        asked_questions_history[question] = 'answered_from_gpt'
        return {"answer": generated_answer}
