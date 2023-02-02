# Disaster-Response
Udacity Data Science Nanodegree Project 2

## Table of Contents
 * [Project Motivation](#project-motivation)
 * [File Descriptions](#file-descriptions)
 * [Components](#components)
 * [Instructions of How to Interact With Project](#instructions-of-how-to-interact-with-project)
 * [Resources](#resources)
 
### Project Motivation
In this project, data engineering skills acquired from the udacity Data Science Nanodegree was applied to analyze disaster data from [Figure Eight](https://appen.com/) to build a model for an API that classifies disaster messages. A machine learning pipeline is created to categorize real messages that were sent during disaster events so that the messages could be sent to an appropriate disaster relief agency. The project includes a web app where an emergency worker can input a new message and get classification results in several categories. The web app will also display visualizations of the data.

### File Descriptions
app    

| - template    
| |- master.html # main page of web app    
| |- go.html # classification result page of web app    
|- run.py # Flask file that runs app    


data    

|- disaster_categories.csv # data to process    
|- disaster_messages.csv # data to process    
|- process_data.py # data cleaning pipeline    
|- DisasterResources.db # database that saved clean data  


models   

|- train_classifier.py # machine learning pipeline     
|- classifier.pkl # saved model     


notebooks   

|- ETL Pipeline Preparation # python notebook used to create ETL pipeline     
|- ML Pipeline Preparation # python notebook used to create MLL pipeline   

README.md    

### Components
Three important components of the project. 

#### 1. ETL Pipeline
*process_data.py* is a python data cleaning pipeline that:

 * Loads the messages and categories datasets
 * Merges the two datasets
 * Cleans the data
 * Stores it in a SQLite database
 
A jupyter notebook *ETL Pipeline Preparation* was used to do EDA to prepare the process_data.py python script. 
 
#### 2. ML Pipeline
*train_classifier.py* is a python machine learning pipeline that:

 * Loads data from the SQLite database
 * Splits the dataset into training and test sets
 * Builds a text processing and machine learning pipeline
 * Trains and tunes a model using GridSearchCV
 * Outputs results on the test set
 * Exports the final model as a pickle file
 
A jupyter notebook *ML Pipeline Preparation* was used to do EDA to prepare the train_classifier.py python script. 

#### 3. Flask Web App
The project includes a web app where an emergency worker can input a new message and get classification results in several categories. The web app will also display visualizations of the data.

### Instructions of How to Interact With Project:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

### Resources
Udacity Data Science Nanodegree Program
