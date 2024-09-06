Drought Prediction using Machine Learning based on datasets available at - https://www.kaggle.com/datasets/cdminix/us-drought-meteorological-data/code

model directory contains all machine learning models used to experiment to find most efficient model in drought prediction
aws code directory contains all files associated with models trained on aws sagemaker and files required for web app deployment

aws models directory includes top 3 models trained and tested on sagemaker instances for local vs cloud runtime comparison
deployment directory contains code to deploy and create and endpoint for prediction and to upload packaged model to s3 buckets

source directory contains the app.py file and inference.py file used to run and host the flask web app and the logic for data preprocessing for the model
templates directory contains the html file for web interface

requirements.txt file mentions the libraries and packages installed additionally on the cloud instance.
