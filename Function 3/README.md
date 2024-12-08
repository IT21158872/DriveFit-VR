# Function 3: Comprehensive Practical Test

## Description
This function conducts a VR-based practical driving test. It tracks applicants' mistakes and predicts future performance based on assessment data.

## Models
- `future_prediction_model.pkl`: Predicts future driving performance based on test results.
- `scaler.pkl`: Used for normalizing data inputs.
- `encoder.pkl`: Encodes categorical features in the dataset.

## Datasets
- `driver_future_performance.csv`: Contains performance metrics for training and evaluation.

## APIs Testing Tutorial
1. Run the `uvicorn prediction_api:app --reload` by navigating to the folder.
2. Open Swagger UI at `http://localhost:8000/docs`.
3. Test the following endpoint:
   - `/predict-future-performance`: Submit test data to receive a prediction on future driving performance.