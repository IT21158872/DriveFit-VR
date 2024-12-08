from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict
import pandas as pd
import pickle
import json

# Initialize FastAPI app
app = FastAPI(title="Question Recommendation API", version="1.0")

with open("Function 2/Models/answer_prediction_model.pkl", "rb") as file:
    trained_model = pickle.load(file)

with open("Function 2/Datasets/quiz_1_features.json", "r") as file:
    quiz_1_features = json.load(file)

with open("Function 2/Datasets/questions.json", "r") as file:
    question_bank = json.load(file)

# Pydantic models for validation
class QuizRequest(BaseModel):
    user_id: str
    quiz_1_features: List[Dict]
    question_bank: List[Dict]
    num_questions: int = 40

class QuizResponse(BaseModel):
    user_id: str
    questions: List[Dict]

# Recommendation logic
def recommend_questions(user_id, quiz_1_features, question_bank, trained_model, num_questions=40):
    quiz_1_df = pd.DataFrame(quiz_1_features)
    question_df = pd.DataFrame(question_bank)

    # Filter user data
    user_data = quiz_1_df[quiz_1_df["user_id"] == user_id]

    # Identify weak categories
    category_performance = user_data.groupby("category")["is_correct"].mean()
    weak_categories = category_performance[category_performance < 0.5].index.tolist()

    # Add derived features
    question_df["overall_accuracy"] = user_data["overall_accuracy"].iloc[0]
    question_df["category_performance"] = question_df["category"].map(
        lambda cat: category_performance.get(cat, 0)
    )

    # Ensure numeric columns
    numeric_columns = ["difficulty", "importance_weight", "overall_accuracy", "category_performance"]
    question_df[numeric_columns] = question_df[numeric_columns].apply(pd.to_numeric, errors="coerce")

    # Predict performance
    features = question_df[numeric_columns]
    question_df["predicted_performance"] = trained_model.predict(features)

    # Filter and rank
    weak_questions = question_df[question_df["category"].isin(weak_categories)]
    recommended_questions = weak_questions.sort_values(
        by=["predicted_performance", "importance_weight"], ascending=[True, False]
    ).head(num_questions)

    # Output question data
    output_columns = ["question_id", "category", "question"]
    return recommended_questions[output_columns].to_dict(orient="records")


# API Endpoints
@app.get("/")
def health_check():
    return {"status": "running", "message": "Question Recommendation API"}

@app.post("/recommend", response_model=QuizResponse)
def recommend_quiz_questions(request: QuizRequest):
    try:
        recommended_questions = recommend_questions(
            user_id=request.user_id,
            quiz_1_features=request.quiz_1_features,
            question_bank=request.question_bank,
            trained_model=trained_model,
            num_questions=request.num_questions
        )
        return QuizResponse(user_id=request.user_id, questions=recommended_questions)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
