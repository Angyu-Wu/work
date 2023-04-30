# %%
# Problem 1 Raw Data Processing
import pandas as pd
import os
import fastparquet as fp
directory = rf'C:\Users\Wu\Desktop\book\stock dataset'
def Raw_Data_Processing():
    ETFs = pd.DataFrame()
    Stocks = pd.DataFrame()
    #Meta = pd.DataFrame()


    for root, dirs, files in os.walk(directory):
        for file in files:
            source = (os.path.join(root, file))
            if (source.split('\\')[-2]) == 'etfs':
                etf = pd.read_csv(os.path.join(root, file))
                etf.insert(0, "Symbol", file.split('.')[0], allow_duplicates=True)
                ETFs = pd.concat([ETFs,etf], ignore_index=True)
            if (source.split('\\')[-2]) == 'stocks':
                stock = pd.read_csv(os.path.join(root, file))
                stock.insert(0, "Symbol", file.split('.')[0], allow_duplicates=True)
                Stocks = pd.concat([Stocks,stock], ignore_index=True)
            if source.endswith('symbols_valid_meta.csv'):
                Meta = pd.read_csv(os.path.join(root, file))

            # etf = pd.read_csv(os.path.join(root, file))
        #for file in files:
        #    source = (os.path.join(root, file))
        #ETF = pd.read_csv('')
    Combined = pd.concat([Stocks,ETFs], ignore_index=True)

    Meta = Meta.drop(columns=['Nasdaq Traded', 'Listing Exchange','Market Category','ETF','Round Lot Size','Test Issue','Financial Status','CQS Symbol','NASDAQ Symbol','NextShares'])

    output = Meta.merge(Combined, how = 'inner', on = ['Symbol'])

    output.astype({
    'Symbol': 'string',
    'Security Name': 'string',
    'Date': 'string',
    'Open': 'float',
    'High': 'float',
    'Low': 'float',
    'Close': 'float',
    'Adj Close': 'float',
    'Volume': 'int'}).dtypes


    parquet_file = directory+r'\Stock_Market_Dataset.parquet'
    fp.write(parquet_file, output, compression = 'GZIP')
    

# %%
#Problem 2: Feature Engineering
def Feature_Engineering():
    output_new = pd.read_parquet(directory+r'\Stock_Market_Dataset.parquet')
    output_new['vol_moving_avg'] = output_new['Volume'].rolling(30).mean()
    output_new['adj_close_rolling_med'] = output_new['Volume'].rolling(30).median()

    # removing all the NULL values using
    # dropna() method
    output_new.dropna(inplace=True,ignore_index=True)
    output_new.to_csv(directory+r'\Stock_Market_Dataset.csv', index=False)
# printing Dataframe



# %% [markdown]
# # Bonus 2 Unit test, manually calculate the mean value and compare with the previous result.
# output = pd.read_parquet(directory+r'\Stock_Market_Dataset.parquet')
# output_new = pd.read_parquet(directory+r'\Stock_Market_Dataset.parquet')
# output_new['vol_moving_avg'] = output_new['Volume'].rolling(30).mean()
# output_new['adj_close_rolling_med'] = output_new['Volume'].rolling(30).median()
# 
# def unit_test(output, output_new):
#     for i in range(0,len(output)):
#         sum = 0
#         for j in range(0,30):
#             sum += output['Volume'][i+j]
#             if output_new['vol_moving_avg'][i] != sum/30:
#                 return False
#     return True

# %%
#Problem 3: Integrate ML Training
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

def Integrate_ML_Training():
    data = pd.read_csv(directory+r'\Stock_Market_Dataset.csv')
    # Assume `data` is loaded as a Pandas DataFrame
    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index('Date', inplace=True)

    # Remove rows with NaN values
    data.dropna(inplace=True)

    # Select features and target
    features = ['vol_moving_avg', 'adj_close_rolling_med']
    target = 'Volume'

    X = data[features]
    y = data[target]

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create a RandomForestRegressor model
    model = RandomForestRegressor(n_estimators=100, random_state=42)

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions on test data
    y_pred = model.predict(X_test)

    # Calculate the Mean Absolute Error and Mean Squared Error
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)


    filename = directory+r'\finalized_model.sav'
    #pickle.dump(model, open(filename, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
    joblib.dump(model, filename)

    log = {'mae': [mae], 'mse': [mse]}
    log1 = pd.DataFrame(data=log)

    directory_log = rf'C:\Users\Wu\Desktop\book\stock dataset\error_log.xlsx'
    log1.to_excel(directory_log)

# %% [markdown]
# # Bonus 3 The model is linear Support Vector Machine with L2 regularization, better accuracy
# from sklearn.svm import LinearSVC
# 
# svc = LinearSVC(C=0.1, penalty='l2').fit(X_train, y_train)

# %%
#Problem 4: Model Serving
from flask import Flask, request, jsonify
import pandas as pd
import ast
def Model_Serving():
    app = Flask(__name__)

    directory = rf'C:\Users\Wu\Desktop\book\stock dataset'
    filename = directory+r'\finalized_model.sav'
    loaded_model = joblib.load(open(filename, 'rb'))

    @app.route('/predict', methods=['GET'])
    def respond():
        
        vol_moving_avg = request.args.get("vol_moving_avg",None)
        adj_close_rolling_med = request.args.get("adj_close_rolling_med", None)
        
        d = [{'vol_moving_avg': vol_moving_avg, 'adj_close_rolling_med': adj_close_rolling_med}]
        #d = [{'vol_moving_avg': 1000, 'adj_close_rolling_med': 100}]
        X = pd.DataFrame(data=d)
        y_predicted = loaded_model.predict(X)
        result = f''+str(y_predicted[0])

        response = {}
        response["Prediction"] = f'vol_moving_avg is  {vol_moving_avg} and adj_close_rolling_med {adj_close_rolling_med} then result is {result}'

        # Return the response in json format
        return jsonify(response)

    @app.route('/')
    def index():
        # A welcome message to test our server
        return "<h1>Welcome to our medium-greeting-api!</h1>"

    if __name__ == '__main__':
        # Threaded option to enable multiple instances for multiple user access support
        app.run(threaded=True, port=5000, debug=True)

# %%
import airflow
import datetime
from airflow import DAG
from airflow.operators.dummy_operator import DummyOperator
from airflow.operators.python_operator import PythonOperator
from airflow.utils.dates import days_ago



#DAG_Raw_Data_Processing = Raw_Data_Processing()
#DAG_Feature_Engineering = Feature_Engineering(Raw_Data_Processing)
#DAG_Integrate_ML_Training = Integrate_ML_Training(Feature_Engineering)
#DAG_Model_Serving = Model_Serving(Feature_Engineering)

dag = DAG(
    "Airflow_Data_Pipeline",
    #default_args = default_args,
    #description="",
    #schedule_interval = datetime.timedelta(days = 1),
    #start_date = start_date,
    #catchup = False,
    #tags=["sdg"],
       
   start_date=days_ago(1),
   schedule_interval=None,
) 

DAG_Raw_Data_Processing  = PythonOperator(
    task_id = 'DAG_Raw_Data_Processing',
    python_callable = Raw_Data_Processing,  
    dag = dag
)
DAG_Feature_Engineering = PythonOperator(
    task_id = 'DAG_Feature_Engineering',
    python_callable = Feature_Engineering,
    dag = dag
)
DAG_Integrate_ML_Training = PythonOperator(
    task_id = 'DAG_Integrate_ML_Training',
    python_callable = Integrate_ML_Training,
    dag = dag
)
DAG_Model_Serving = PythonOperator(
    task_id = 'DAG_Model_Serving',
    python_callable = Model_Serving,
    dag = dag
)
ready = DummyOperator(task_id = 'ready')

DAG_Raw_Data_Processing >> DAG_Feature_Engineering >>DAG_Integrate_ML_Training>>DAG_Model_Serving>>ready

# %%



