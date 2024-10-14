Drought Prediction Using Machine Learning
Overview
This project focuses on developing a machine learning model to predict drought occurrences based on historical weather and soil data. By leveraging machine learning techniques, the model aims to predict drought conditions in various regions to aid in early warning systems, improve agricultural planning, and better water resource management.

Project Objectives
The main goals of this project are:
1. To analyze historical weather and soil data and identify key factors contributing to drought conditions.
2. To build and train a machine learning model that can accurately predict droughts based on these factors.
3. To deploy the model on AWS SageMaker for efficient and scalable predictions.

Technologies Used

Programming Language: Python
Libraries:
  Pandas, NumPy for data manipulation and analysis.
  Scikit-learn for model development (Random Forest Classifier).
  Matplotlib, Seaborn for data visualization.
Model Deployment: AWS SageMaker
Database: Amazon S3 for storing and retrieving datasets.
Dataset
  The project utilizes two key datasets:

  Weather Data: Historical weather information, including variables such as precipitation, temperature, humidity, and wind speed.
  Soil Data: Soil moisture levels, soil type, and water retention data relevant to the regions studied.
  Both datasets were preprocessed to handle missing data and combined for better feature representation.

Model Development Process
Data Preprocessing:
Data was cleaned by handling missing values, normalizing the features, and performing feature extraction to derive additional features such as month, year, and day from the dates.
Data integration was carried out by merging the weather dataset with the soil dataset using a common key (region identifier).

Feature Engineering:
Relevant features were selected based on domain knowledge and feature selection methods like Recursive Feature Elimination (RFE).
Features such as temperature, precipitation, humidity, soil moisture, and fips code were identified as important factors in predicting drought conditions.
Model Training:

A Random Forest Classifier was used for training the model due to its ability to handle large datasets and provide feature importance insights.
The dataset was split into training and testing sets for model evaluation. Hyperparameter tuning was performed to enhance the model’s accuracy.

Model Evaluation:
The model was evaluated using metrics like accuracy, precision, recall, and F1-score.
Confusion matrices were generated to better understand the model’s performance in predicting drought occurrences.

Deployment
The trained model was deployed on AWS SageMaker for real-time predictions. SageMaker was chosen for its scalable infrastructure, allowing for efficient model deployment and monitoring.
API Endpoint: An endpoint was created using AWS SageMaker to enable real-time predictions on new data.

Results
The model achieved an accuracy of X% on the test dataset, with significant precision and recall metrics.
Feature importance analysis revealed that precipitation and soil moisture levels were the most critical predictors of drought conditions.

Drought Prediction using Machine Learning based on datasets available at - https://www.kaggle.com/datasets/cdminix/us-drought-meteorological-data/code

model directory contains all machine learning models used to experiment to find most efficient model in drought prediction
aws code directory contains all files associated with models trained on aws sagemaker and files required for web app deployment

aws models directory includes top 3 models trained and tested on sagemaker instances for local vs cloud runtime comparison
deployment directory contains code to deploy and create and endpoint for prediction and to upload packaged model to s3 buckets

source directory contains the app.py file and inference.py file used to run and host the flask web app and the logic for data preprocessing for the model
templates directory contains the html file for web interface

requirements.txt file mentions the libraries and packages installed additionally on the cloud instance.
