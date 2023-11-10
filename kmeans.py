import pickle
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

MiniBatchKMeans_model = pickle.load(open("MiniBatchKMeans_model.pkl", 'rb'))

def MiniBatchKMeans_predict(age, job, marital, education, default, balance, housing, loan, contact, day, month, duration, campaign,
                pdays=-1, previous=0):
    print("Started predicting")
    data = pd.DataFrame({
        'age': [age],
        'job': [job],
        'marital': [marital],
        'education': [education],
        'default': [default],
        'balance': [balance],
        'housing': [housing],
        'loan': [loan],
        'contact': [contact],
        'day': [day],
        'month': [month],
        'duration': [duration],
        'campaign': [campaign],
        'pdays': [pdays],
        'previous': [previous]
    })
    print(data)
    print("Ended predicting")
    return MiniBatchKMeans.predict(data)


print(log_predict(1, "student", "single", "secondary", "yes", 100, "no", "no", "cellular", 1, "oct", 365, 0, -1, 0))

