# Disaster Response Pipeline Project

### Project Overview:
The target of this project is to analyze the disaster data from Figure Eight, then build a ML model which can classifies disaster messages. 

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/  
If you run the Web App from the Udacity Workspace IDE, to access the web page please follow the steps below:  
* open another Terminal 
* type `env|grep WORK` You'll see output that looks something like this: view6914b2f4
* then access: https://view6914b2f4-3001.udacity-student-workspaces.com (Your SPACEID might be different) 

### Files:
<pre>

- app
| - template
| |- master.html  # main page of web app
| |- go.html  # classification result page of web app
|- run.py  # Flask file that runs app

- data
|- disaster_categories.csv  # data to process 
|- disaster_messages.csv  # data to process
|- process_data.py
|- InsertDatabaseName.db   # database to save clean data to

- models
|- train_classifier.py
|- classifier.pkl  # saved model 

- ETL Pipeline Preparation.ipynb  # ETL Notebook

- ML Pipeline Preparation.ipynb  # ML Notebook

- README.md

</pre>

<a id='sw'></a>
