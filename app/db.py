# app/db.py

import mysql.connector
from config.settings import settings

def connect_to_db():
    connection = mysql.connector.connect(
        host=settings.DB_HOST,
        user=settings.DB_USER,
        password=settings.DB_PASSWORD,
        database=settings.DB_NAME
    )
    return connection

def get_answers_from_db(question: str):
    connection = connect_to_db()
    cursor = connection.cursor()
    # query = "SELECT answer FROM qa_pairs WHERE question = %s"
    # cursor.execute(query, (question,))
    query = "SELECT answer FROM qa_pairs WHERE LOWER(question) LIKE %s"
    cursor.execute(query, (question.strip().lower(),))
    answers = cursor.fetchall()
    cursor.close()
    connection.close()
    return answers
