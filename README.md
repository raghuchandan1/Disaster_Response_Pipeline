# Disaster Response Pipeline Project
### Getting Started
Lots of messages are generated during a disaster requesting for help and support. Due to the overwhelming number of messages which cannot be parsed by a human such may just be stacked without any reach. So if we can classify which type of support the messages points to, then that message can be segregated and forwarded to the respective department and the NGOs dealing with that category and help can be provided much quickly. This project does the same using NLP. 
The project classifies a given message into one of the 36 disaster management categories. The goal is to classify the message under the correct category so that the message can forwarded to the right target.
The project includes a web app where a message from a disaster can be given as input and the web app produces the output showing which categories the message is related to.

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
