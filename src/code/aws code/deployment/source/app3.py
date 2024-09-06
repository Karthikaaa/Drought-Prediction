from flask import Flask, request, render_template, jsonify
import boto3
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import io
import sys
import os
# import openpyxl

print(sys.executable)
print(sys.path)


# # Function to install a package dynamically
# def install(package):
#     subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# # Check if openpyxl is installed, if not, install it
# try:
#     import openpyxl
# except ImportError:
#     install('openpyxl')

# os.system('pip install openpyxl')

# os.system('pip install fsspec')

app = Flask(__name__)

# Initialize the SageMaker Runtime client
sagemaker_runtime = boto3.client('runtime.sagemaker', region_name='eu-north-1')
sagemaker_client = boto3.client('sagemaker', region_name='eu-north-1')

# Endpoint details

ENDPOINT_NAME = 'best-random-forest-endpoint'


@app.route('/')
def index():
    return render_template('formnew.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"})

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"})

    # Check if the file is a CSV
    if file and file.filename.endswith('.csv'):
        try:
            # Load the CSV data into a DataFrame
            testset = pd.read_csv(file)

            # Create an in-memory binary stream to save the CSV file
            csv_buffer = io.StringIO()
            # Save the DataFrame back to a CSV file in memory
            testset.to_csv(csv_buffer, index=False)

            # Reset the buffer's position to the beginning
            csv_buffer.seek(0)

            # Create the SageMaker runtime client
            sagemaker_runtime = boto3.client('sagemaker-runtime')

            try:
                # Invoke the SageMaker endpoint
                response = sagemaker_runtime.invoke_endpoint(
                    EndpointName=ENDPOINT_NAME,
                    ContentType='text/csv',  # Set content type to CSV MIME type
                    Body=csv_buffer.getvalue()  # Read the CSV content from the buffer and send it to the endpoint
                )

                result = response['Body'].read().decode('utf-8')
                return jsonify({"result": result, "message": "Form submitted and model is running..."})
            except sagemaker_runtime.exceptions.ModelError as e:
                error_message = e.response['Error']['Message']
                print(f"Model Error: {error_message}")
                return jsonify({"error": f"Error calling endpoint: {error_message}"})
            except Exception as e:
                print(f"General Error: {str(e)}")
                return jsonify({"error": f"Error calling endpoint: {str(e)}"})
        except Exception as e:
            return jsonify({"error": f"Error reading CSV file: {str(e)}"})
    else:
        return jsonify({"error": "Unsupported file format. Please upload a CSV file."})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80, debug=True)
