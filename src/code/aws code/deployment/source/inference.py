import os
os.system('pip install fsspec')
os.system('pip install s3fs')

import json
import s3fs
import boto3

import pandas as pd
import joblib  # or pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import io
import sys
import boto3

# Load the model
def model_fn(model_dir):
    model = joblib.load(os.path.join(model_dir, "bestfit_random_forest_model.pkl"))
    return model

# Handle input data
# def input_fn(request_body, request_content_type):
#     if request_content_type == 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet':
#         # Load Excel file into a DataFrame
#         df = pd.read_excel(request_body)
#         return df
#     else:
#         raise ValueError("Unsupported content type: {}".format(request_content_type))



def input_fn(request_body, request_content_type):
    if request_content_type == 'text/csv':
        try:
            # Convert request_body (which is bytes) to a file-like object
            # csv_file = io.StringIO(request_body.decode('utf-8'))  # Decode bytes to string for CSV
            csv_file = io.StringIO(request_body)
            # Load CSV file into a DataFrame
            df = pd.read_csv(csv_file)
            return df
        except Exception as e:
            raise ValueError(f"Error reading CSV file: {e}")
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")



# Make predictions and calculate performance metrics
def predict_fn(input_data, model):

    # Pre-load the soil dataset
    print(sys.executable)
    print(sys.path)
    #fs = s3fs.S3FileSystem()
    # Pre-load the soil dataset
    # soilset = pd.read_csv('s3://sagemaker-studio-637423247598-ibwjh7qnlsc/datasets/soil_data.csv')
    s3_client = boto3.client('s3')
    
    try:
        # Read soil data from S3
        soil_response = s3_client.get_object(Bucket='sagemaker-studio-637423247598-ibwjh7qnlsc', Key='datasets/soil_data.csv')
        
        # Load the CSV file into a DataFrame
        soilset = pd.read_csv(io.BytesIO(soil_response['Body'].read()))
        print("soil dataset read")
    
    except Exception as e:
        print(f"Error reading soil data from S3: {e}")
        raise ValueError(f"Error reading soil data from S3: {e}")
    
    
    # s3_file_path = 's3://sagemaker-studio-637423247598-ibwjh7qnlsc/datasets/train_timeseries.csv'
    # chunksize = 10000
    # chunk_list = []

    # try:
    #     # Read the CSV file in chunks and append each chunk to the list
    #     for chunk in pd.read_csv(s3_file_path, chunksize=chunksize):
    #         chunk_list.append(chunk)

    #     # Concatenate all chunks into a single DataFrame
    #     weathertrainset = pd.concat(chunk_list, ignore_index=True)
    # except Exception as e:
    #     print(f"An error occurred: {e}")

    # Read weather data from S3 in chunks
    # chunksize = 10000
    # chunk_list = []

    # try:
    #     weather_response = s3_client.get_object(Bucket='sagemaker-studio-637423247598-ibwjh7qnlsc', Key='datasets/train_timeseries.csv')
    #     weather_chunks = pd.read_csv(io.BytesIO(weather_response['Body'].read()), chunksize=chunksize)
        
    #     for chunk in weather_chunks:
    #         chunk_list.append(chunk)
        
    #     weathertrainset = pd.concat(chunk_list, ignore_index=True)
    #     print(" Weather train dataset read successfully.")
    # except Exception as e:
    #     raise ValueError(f"Error reading weather data from S3: {e}")

    chunksize = 10000
    chunk_list = []
    
    try:
        weather_response = s3_client.get_object(Bucket='sagemaker-studio-637423247598-ibwjh7qnlsc', Key='datasets/train_timeseries.csv')
        weather_chunks = pd.read_csv(io.BytesIO(weather_response['Body'].read()), chunksize=chunksize)
        
        for chunk in weather_chunks:
            chunk_list.append(chunk)
        
        # Concatenate all chunks into a single DataFrame
        weathertrainset = pd.concat(chunk_list, ignore_index=True)
        print("Weather train dataset read successfully.")
    
    except Exception as e:
        raise ValueError(f"Error reading weather data from S3: {e}")
    
    # print("Data reading complete.")





    

    print("train dataset read")

    weathertrainset = weathertrainset.dropna()
    weathertrainset['month'] = pd.DatetimeIndex(weathertrainset['date']).month
    weathertrainset['year'] = pd.DatetimeIndex(weathertrainset['date']).year
    weathertrainset['day'] = pd.DatetimeIndex(weathertrainset['date']).day
    print("weathertrainset date extracted")

    train_merged_df = pd.merge(weathertrainset, soilset, on='fips')
    print("weathertrainset and soilset merged")

    train_merged_df = train_merged_df.dropna()
    train_merged_df['score'] = train_merged_df['score'].astype(int)
    print("weathertrainset null dropped")

    # Load and preprocess training data for feature selection
    scaler = StandardScaler()
    X_train = train_merged_df.drop(columns=['score'])
    y_train = train_merged_df['score']
    X_train = X_train.drop(columns=['date']).dropna()
    print("train date dropped")

    X_train_scaled = scaler.fit_transform(X_train)
    print("train scaled")

    # Perform feature selection
    modelrf = RandomForestClassifier(n_estimators=10, random_state=42, n_jobs=-1)
    subset_size = 50000
    X_train_subset, _, y_train_subset, _ = train_test_split(X_train_scaled, y_train, train_size=subset_size, random_state=42, stratify=y_train)
    rfe = RFE(modelrf, n_features_to_select=15)
    fit = rfe.fit(X_train_subset, y_train_subset)
    selected_indices = np.where(fit.support_)[0]
    print("indices selected")

    input_data = input_data.dropna()  # Drop missing values
    input_data['month'] = pd.DatetimeIndex(input_data['date']).month
    input_data['year'] = pd.DatetimeIndex(input_data['date']).year
    input_data['day'] = pd.DatetimeIndex(input_data['date']).day

    # Merge with soil data
    test_merged_df = pd.merge(input_data, soilset, on='fips')
    test_merged_df['score'] = test_merged_df['score'].astype(int)
    print("test and soil merged")

    # Prepare features and target
    X_test = test_merged_df.drop(columns=['score'])
    y_true = test_merged_df['score']
    X_test = X_test.drop(columns=['date']).dropna()
    print("test date dropped")

    # Scale the features
    X_test_scaled = scaler.transform(X_test)
    print("test scaled")

    # Select features based on RFE
    X_test_scaled = X_test_scaled[:, selected_indices]
    print("test fitted with selected indices")

    # Predict using the model
    y_pred = model.predict(X_test_scaled)
    print("predictions are made")

    # Calculate performance metrics
    accuracy = accuracy_score(y_true, y_pred)
    print("accuracy calculated")
    precision_macro = precision_score(y_true, y_pred, average='macro')
    print("precision_macro calculated")
    recall_macro = recall_score(y_true, y_pred, average='macro')
    print("recall_macro calculated")
    f1_macro = f1_score(y_true, y_pred, average='macro')
    print("f1_macro calculated")
    precision_micro = precision_score(y_true, y_pred, average='micro')
    print("precision_micro calculated")
    recall_micro = recall_score(y_true, y_pred, average='micro')
    print("recall_micro calculated")
    f1_micro = f1_score(y_true, y_pred, average='micro')
    print("f1_micro calculated")
    conf_matrix = confusion_matrix(y_true, y_pred).tolist()  # Convert to list for JSON serialization
    print("conf_matrix calculated")

    # Return metrics rounded to 4 decimal places
    result = {
        "accuracy": round(accuracy, 4),
        "precision_macro": round(precision_macro, 4),
        "recall_macro": round(recall_macro, 4),
        "f1_macro": round(f1_macro, 4),
        "precision_micro": round(precision_micro, 4),
        "recall_micro": round(recall_micro, 4),
        "f1_micro": round(f1_micro, 4),
        "confusion_matrix": conf_matrix
    }
    print("Result:", result)
    print("result is being returned next")
    return result
    # print("result returned")

# Format the output
def output_fn(prediction, content_type):
    return json.dumps(prediction)

