# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from sklearn.preprocessing import LabelEncoder
import json
import pickle

app = Flask(__name__)

# Load the model
model = pickle.load(open('gradient_boosting_model.pkl','rb'))

def get_record_dummy(keys, values):    
    dataframe = pd.DataFrame(keys, columns = ['columns'], index = keys)
    dataframe_dummy = pd.get_dummies(dataframe)
    list_value = dataframe_dummy.loc[values].tolist()
    return list_value

def get_label_value(keys, value):
    le = LabelEncoder()
    dataframe = pd.DataFrame(keys, columns = ['columns'], index = keys)
    dataframe['columns'] = le.fit_transform(dataframe['columns'])
    values = int(dataframe.loc[value])
    return values

def convert(x):
    if x == 1:
        return "yes"
    else:
        return "no"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    df = pd.DataFrame(data, columns = ['age', 'job', 'marital', 'education',
                                       'defaults', 'balance', 'housing', 'loan',
                                       'contact', 'day', 'month', 'duration',
                                       'campaign', 'poutcome'])
    
    array_df = []
    
    for subscript in range(0, len(df)):
        age = df['age'].loc[subscript]
        
        balance = df['balance'].loc[subscript]
        
        duration = df['duration'].loc[subscript]
        
        campaign = df['campaign'].loc[subscript]
        
        job = get_record_dummy(['admin.', 'technician', 'services', 'management',
                                'retired', 'blue-collar', 'unemployed',
                                'entrepreneur', 'housemaid', 'unknown',
                                'self-employed', 'student'], 
                                df['job'].loc[subscript])
        
        marital = get_record_dummy(['married', 'single', 'divorced'], df['marital'].loc[subscript])
        
        education = get_record_dummy(['secondary', 'tertiary', 'primary', 'unknown'], 
                                   df['education'].loc[subscript])
        
        defaults = get_label_value(['no', 'yes'], df['defaults'].loc[subscript])
        
        housing = get_label_value(['no', 'yes'], df['housing'].loc[subscript])
        
        loan = get_label_value(['no', 'yes'], df['loan'].loc[subscript])
        
        contact = get_record_dummy(['unknown', 'cellular', 'telephone'], 
                                   df['contact'].loc[subscript])
        
        day = get_record_dummy(['1', '2', '3', '4', '5', '6', '7', '8', '9',
                                '10', '11', '12', '13', '14', '15', '16', '17',
                                '18', '19', '20', '21', '22', '23', '24', '25',
                                '26', '27', '28', '29', '30', '31'], 
                                   df['day'].loc[subscript])
        
        month = get_record_dummy(['may', 'jun', 'jul', 'aug', 'oct', 'nov', 'dec',
                                  'jan', 'feb', 'mar', 'apr', 'sep'], 
                                   df['month'].loc[subscript])
        
        poutcome =  get_record_dummy(['unknown', 'other', 'failure', 'success'], 
                                   df['poutcome'].loc[subscript])
        
        lists = [age, defaults, balance, housing, loan, duration, campaign, job, 
                 marital, education, contact, day, month, poutcome]
        
        arrays = []
        for i in lists:
            arrays = np.append(arrays, i)
        
        array_df.append(arrays)
        
    test_set = pd.DataFrame(array_df, columns = ['age', 'defaults', 'balance', 'housing', 'loan', 'duration', 'campaign',
       'job_admin.', 'job_blue-collar', 'job_entrepreneur', 'job_housemaid',
       'job_management', 'job_retired', 'job_self-employed', 'job_services',
       'job_student', 'job_technician', 'job_unemployed', 'job_unknown',
       'marital_divorced', 'marital_married', 'marital_single',
       'education_primary', 'education_secondary', 'education_tertiary',
       'education_unknown', 'contact_cellular', 'contact_telephone',
       'contact_unknown', 'day_1', 'day_10', 'day_11', 'day_12', 'day_13',
       'day_14', 'day_15', 'day_16', 'day_17', 'day_18', 'day_19', 'day_2',
       'day_20', 'day_21', 'day_22', 'day_23', 'day_24', 'day_25', 'day_26',
       'day_27', 'day_28', 'day_29', 'day_3', 'day_30', 'day_31', 'day_4',
       'day_5', 'day_6', 'day_7', 'day_8', 'day_9', 'month_apr', 'month_aug',
       'month_dec', 'month_feb', 'month_jan', 'month_jul', 'month_jun',
       'month_mar', 'month_may', 'month_nov', 'month_oct', 'month_sep',
       'poutcome_failure', 'poutcome_other', 'poutcome_success',
       'poutcome_unknown'])
            
    result = model.predict(test_set)
    
    df['result_prediction'] = result
    df['response'] = df['result_prediction'].apply(lambda x: convert(x))
    
    json_records = df[['age', 'job', 'marital', 'education', 
                      'defaults', 'balance', 'housing', 'loan', 
                      'contact', 'day', 'month', 'duration', 
                      'campaign', 'poutcome', 'response']].to_json(orient ='records')
    
    return json_records

if __name__ == '__main__':
    app.run(port=5000, debug=True)
        