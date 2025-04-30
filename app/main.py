
from fastapi import FastAPI
from random import choice
from app.db import get_answers_from_db
from app.gptj_model import gptj

app = FastAPI()

# Function to generate a new answer using GPT
def generate_gpt_answer(question: str):
    # Send the question directly to GPT for a new, fresh answer
    generated_answer = gptj.generate_answer(question)
    return generated_answer

# Dictionary to track the history of questions asked by users
asked_questions_history = {}

@app.get("/")
async def root():
    return {"Message":"Welcome to the AI Chatbot"}

@app.get("/answer")
async def get_answer(question: str):
    # First, check if the question has been asked before and store it
    if question in asked_questions_history:
        # If the same question has been asked before, always generate a new answer using GPT (no DB lookup)
        generated_answer = generate_gpt_answer(question)
        return {"answer": generated_answer}
    
    # If it's the first time the question is asked, check if the answer exists in DB
    normalized_question = question.strip().lower()
    answers = get_answers_from_db(question)

    if answers:  # If there are answers in the DB
        # Fetch all answers from DB (assuming get_answers_from_db returns a list of tuples)
        sampled_answers = [answer[0] for answer in answers]

        # Randomly choose one answer from the DB and return it
        selected_answer = choice(sampled_answers)
        asked_questions_history[question] = 'answered_from_db'  # Mark that this question was answered from DB
        
        return {"answer": selected_answer}
    else:
        # If no answer is found in the DB, generate an answer using GPT
        generated_answer = generate_gpt_answer(question)
        asked_questions_history[question] = 'answered_from_gpt'  # Mark that GPT answered this question
        
        return {"answer": generated_answer}
