# Resume and Job Description Matching using Machine Learning

## Overview
This project explores how natural language processing and classical machine learning
can be used to assess similarity between resumes and job descriptions. The goal is to
build an end-to-end ML pipeline that transforms unstructured text into meaningful
features and evaluates model performance.

## Motivation
I built this project independently to strengthen my understanding of applied machine
learning, particularly text preprocessing, feature engineering, and model evaluation.
The problem is closely aligned with real-world hiring and opportunity matching systems.

## Dataset
A small labeled dataset was created manually for learning purposes. Each row contains:
- Resume text
- Job description text
- A binary label indicating match quality

## Approach
- Text preprocessing using tokenization and stopword removal
- TF-IDF vectorization
- Logistic Regression as a baseline model
- Evaluation using accuracy and confusion matrix

## Results
The baseline model achieved reasonable performance given the small dataset. The project
demonstrates the full ML lifecycle rather than optimizing for raw accuracy.

## Future Work
- Larger dataset
- Semantic embeddings
- Improved evaluation metrics
