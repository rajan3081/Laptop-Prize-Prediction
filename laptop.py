import numpy as np
import pandas as pd
from warnings import filterwarnings
filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, ExtraTreesRegressor
from sklearn.svm import SVR
import os
if os.path.exists('df.pkl'):
    df = pd.read_pickle('df.pkl')
    print('Loaded processed df from df.pkl')
else:
    df = pd.read_csv(r"C:\Users\lenovo\Downloads\laptop_data.csv")
# prepare features & labels
df.head(1)
X = df.drop(columns=['Price'])
y = np.log(df['Price'])
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.15,random_state=2)
df.shape
df.info()
df.duplicated().sum()
df.isnull().sum()
df.describe()
df.columns
if 'Unnamed: 0' in df.columns:
    df.drop(columns=['Unnamed: 0'], inplace=True)
df.head()
if 'Ram' in df.columns and df['Ram'].dtype == object:
    df['Ram']=df['Ram'].str.replace('GB','')
if 'Weight' in df.columns and df['Weight'].dtype == object:
    df['Weight']=df['Weight'].str.replace('kg','')
df.head(1)

#Exporting the Model
# --- Train models in a loop and save metrics/pipelines ---
import pickle, json, os

models = [
    ('LinearRegression', LinearRegression()),
    ('Ridge', Ridge(alpha=10)),
    ('Lasso', Lasso(alpha=0.001)),
    ('KNN', KNeighborsRegressor(n_neighbors=3)),
    ('DecisionTree', DecisionTreeRegressor(max_depth=8)),
    ('SVR', SVR(kernel='rbf',C=10000,epsilon=0.1)),
    ('RandomForest', RandomForestRegressor(n_estimators=100,random_state=3,max_samples=0.5,max_features=0.75,max_depth=15)),
    ('ExtraTrees', ExtraTreesRegressor(n_estimators=100,random_state=3,max_features=0.75,max_depth=15)),
    ('AdaBoost', AdaBoostRegressor(n_estimators=15,learning_rate=1.0)),
    ('GradientBoost', GradientBoostingRegressor(n_estimators=500))
]

metrics = {}
best_score = -999
best_pipe = None
best_name = None
os.makedirs('models', exist_ok=True)
for name, estimator in models:
    transformer = ColumnTransformer(transformers=[
        ('col_tnf', OneHotEncoder(sparse_output=False, drop='first'), [0, 1, 7, 10, 11])
    ], remainder='passthrough')
    pipe_model = Pipeline([
        ('transformer', transformer),
        ('estimator', estimator)
    ])
    pipe_model.fit(X_train, y_train)
    y_pred = pipe_model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    metrics[name] = {'r2_score': float(r2), 'mae': float(mae)}
    with open(os.path.join('models', name + '.pkl'), 'wb') as f:
        pickle.dump(pipe_model, f)
    print(f"Trained {name}: R2={r2:.4f}, MAE={mae:.4f}")
    if r2 > best_score:
        best_score = r2
        best_pipe = pipe_model
        best_name = name

with open('models/metrics.json', 'w') as f:
    json.dump(metrics, f, indent=4)

pickle.dump(df, open('df.pkl', 'wb'))
print('Saved df and all trained models')
if best_pipe is not None:
    pickle.dump(best_pipe, open('pipe.pkl', 'wb'))
    print(f'Saved best model as pipe.pkl: {best_name}')


#Exporting the Model

import pickle

pickle.dump(df,open('df.pkl','wb'))
print(df)
 