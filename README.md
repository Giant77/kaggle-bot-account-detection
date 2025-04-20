# Kaggle Bot Account Detection

This project was developed as part of the Machine Learning Practical Mid-Semester Exam. It aims to classify Kaggle accounts as either bot or legitimate based on user features like follower count, gender, number of uploaded datasets, and various other attributes.

## Project Overview

The project involves training a machine learning model to detect bot accounts on Kaggle, saving the trained model, and integrating it with a web application that can receive user input data and provide classification predictions.

## Team Members (Group 10)

- Willy Jonathan Arsyad
- Muhammad Khalid Al Ghifari
- Iwani Khairina

## Dataset

The Kaggle Bot Account Detection dataset contains information about fake or bot accounts on the Kaggle platform. It includes various features that can be used to analyze and identify patterns of bot account behavior. More information about this dataset can be found at [Kaggle Bot Account Detection Dataset](https://www.kaggle.com/datasets/shriyashjagtap/kaggle-bot-account-detection).

## Project Structure

```
.
├── README.md
├── REF_kaggle-bot-account-detection.ipynb
├── application
│   ├── backend
│   │   ├── __init__.py
│   │   ├── main.py
│   │   └── requirements.txt
│   └── frontend
│       ├── main.py
│       └── requirements.txt
├── deskripsi UTS.pdf
├── main.py
├── model
│   ├── DT_model.pkl
│   ├── ET_model.pkl
│   ├── SVM_model.pkl
│   ├── logreg_model.pkl
│   └── minmax_scaler.pkl
├── notebook.ipynb
└── requirements.txt
```

- **application**: Contains all code needed to run the application
  - **backend**: Contains FastAPI implementation for the API
  - **frontend**: Contains Streamlit implementation for the user interface
- **model**: Stores the trained machine learning models ready for prediction
- **notebook.ipynb**: Jupyter notebook used for data exploration, preprocessing, and model training

## Key Features

1. **Machine Learning Model Training**

   - Exploratory Data Analysis (EDA)
   - Data preprocessing (handling duplicates, missing values)
   - Feature correlation analysis
   - Handling imbalanced dataset
   - Preprocessing pipeline (encoding categorical data, scaling numeric data)

2. **Model Optimization**

   - Hyperparameter tuning
   - Cross-validation
   - Ensemble learning techniques

3. **Web Application**
   - Backend using FastAPI
   - Frontend using Streamlit
   - Consistent preprocessing pipeline between training and prediction

## How to Run

### Prerequisites

- Python 3.x
- Required Python packages listed in requirements.txt files

### Setting Up the Backend

1. Navigate to the backend directory:

   ```
   cd application/backend
   ```

2. Install the required dependencies:

   ```
   pip install -r requirements.txt
   ```

3. Start the FastAPI server:
   ```
   python main.py
   ```

### Setting Up the Frontend

1. Navigate to the frontend directory:

   ```
   cd application/frontend
   ```

2. Install the required dependencies:

   ```
   pip install -r requirements.txt
   ```

3. Start the Streamlit application:
   ```
   streamlit run main.py
   ```

## Important Notes

- The preprocessing pipeline implemented during model training must be identically implemented in the backend pipeline.
- Features used for prediction must match those used during model training.
- Make sure both backend and frontend are running simultaneously for the application to work correctly.

## Evaluation Metrics

The project is evaluated based on:

- Comprehensive EDA and preprocessing implementation (10 points)
- Implementation of hyperparameter tuning and cross-validation (20 points)
- Implementation of ensemble learning techniques (20 points)
- Completion of all required code (20 points)
- Successful execution of the entire application (30 points)
