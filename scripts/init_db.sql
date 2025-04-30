-- scripts/init_db.sql

CREATE DATABASE IF NOT EXISTS chatbot_db;

USE chatbot_db;

CREATE TABLE IF NOT EXISTS qa_pairs (
    id INT AUTO_INCREMENT PRIMARY KEY,
    question TEXT NOT NULL,
    answer TEXT NOT NULL
);


-- INSERT INTO qa_pairs (question, answer) VALUES
-- ('What is Python?', 'Python is a high-level, interpreted programming language.'),
-- ('What is AI?', 'AI stands for Artificial Intelligence, which refers to the simulation of human intelligence in machines.');
