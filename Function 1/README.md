# Function 1: Sensory Readiness Evaluation

## Description
This function assesses the eyesight and hearing capabilities of driving school applicants using VR technology. It categorizes sensory levels (Good, Normal, Bad) and generates reports to evaluate sensory readiness for safe driving.

## Models
- `sound_localization_model.pkl`: Predicts hearing capabilities based on synthetic and real-world data.
- `visual_acuity_model.pkl`: Predicts eyesight levels using visual distance thresholds.

## Datasets
- `sound_localization_data.csv`: Contains synthetic and real-world data for training the sound localization model.
- `visual_acuity_data.csv`: Contains data for training the eyesight assessment model.
- `synthetic_data_generation.ipynb`: Notebook used to generate synthetic datasets.

## APIs Testing Tutorial
1. Run the `uvicorn hearing_test_api:app --reload` and `uvicorn visual_test_api:app --reload` APIs by navigating to the folders.
2. Navigate to the Swagger UI at `http://localhost:8000/docs` (adjust the port if necessary).
3. Use the following endpoints for testing:
   - `/hearing-test`: Test hearing assessment by providing sample input data.
   - `/visual-test`: Test eyesight assessment by uploading input parameters.