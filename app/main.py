from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from random import choice
from app.db import get_answers_from_db
from app.gptj_model import gptj

app = FastAPI()

# Apply CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this as needed for specific domains
    allow_methods=["*"],
    allow_headers=["*"],
)

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
async def get_answer(request: Request):
    data = await request.json()
    question = data.get("question")

    if not question:
        return {"error": "Question not provided"}

    # Normalize question for consistent processing
    normalized_question = question.strip().lower()

    # Check in DB first
    answers = get_answers_from_db(normalized_question)

    if answers:
        # Fetch all answers from DB (assuming get_answers_from_db returns a list of tuples)
        sampled_answers = [answer[0] for answer in answers]
        selected_answer = choice(sampled_answers)
        asked_questions_history[question] = 'answered_from_db'
        return {"answer": selected_answer}

    # If not found in DB, generate using GPT
    generated_answer = generate_gpt_answer(question)
    asked_questions_history[question] = 'answered_from_gpt'
    return {"answer": generated_answer}
