import joblib
import pandas as pd
# from sklearn.preprocessing import StandardScaler, OneHotEncoder
# from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
# from sklearn.impute import SimpleImputer
model = joblib.load("kaggel_salary_Linear_Regression.pkl")
print('Model Laod Done')
data_dict = {'experience':[],
            'country':[],
            'education':[],
            "languages":[],
            'frameworks':[],
            'company_size':[]}
li = ['experience',	'country',	'education','languages',	'frameworks','company_size']
for i in li:
    j = input(i)
    j = j.strip()
    if j.isnumeric():
        data_dict[i].append(int(j))
    else:
        data_dict[i].append(j)
print(data_dict)
df = pd.DataFrame(data_dict)
# print(df)
print(model.predict(df))
