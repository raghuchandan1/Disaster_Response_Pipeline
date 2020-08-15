# Disaster Response Pipeline Project
### Getting Started
The project tries to classify data into one of the 36 categories used. It tries to classify the message under the correct group so that the message can reach the right target.
The project also includes a web app where a message from a disaster can be given as input and the web app produces the output showing which categories the message is related to.

### Dependencies
- scikit-learn
- numpy
- pandas
- pickle
- nltk
- sqlalchemy

### Files
app:
    - templates: The HTML templates rendered in the web app
    - app.py: Code for the Flask web application
data:
    - disaster_categories.csv: Data file containing the details of the categories
    - disaster_messages.csv: Data containing the messages and the related categories
    - DisasterResponse.db: SQLite database created after cleaning the data
    - process_data.py: Code for cleaning the data
model:
    - classifier.pkl: Model stored after training the pipeline used by the web app for prediction
    - train_classifier.py: Trains a machine learning pipeline

### Setup Instructions
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
