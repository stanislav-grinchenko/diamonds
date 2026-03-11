
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import GridSearchCV

# ----------------------------------
# load dataset
df_diamonds = sns.load_dataset('diamonds')

# ---------------------------------

# data cleaning
def keep_not_null(row) :
    if 0 in row.values : return False
    return True

df_clean = df_diamonds[df_diamonds.apply(keep_not_null,axis=1)]

# ------------------------------------

# data splitting
X = df_clean.drop(columns=["price"])
y = df_clean["price"]

X_train, X_test, y_train, y_test  = train_test_split(X,y, random_state=42)
X_train.shape, X_test.shape, y_train.shape, y_test.shape

# --------------------------------------
# data preprocessing
df_cat = df_clean.select_dtypes(include="category")

cat_pipe = Pipeline(
    [ ("cat_imp",SimpleImputer(strategy="most_frequent"))
      ,("ohe",OneHotEncoder(drop="first",sparse_output=False))
        ])
cat_pipe

num_pipe = Pipeline(
    [("knn_imp", KNNImputer(n_neighbors=5))
     ,("scaler", StandardScaler())
      ])
num_pipe

# ------------------------------------------------------------------
### Final Preprocessor

preprocessor = ColumnTransformer(
    [("numeric",num_pipe, make_column_selector(dtype_include="number"))
    ,("categorical", cat_pipe, make_column_selector(dtype_exclude="number"))
      ]).set_output(transform="pandas")

# ------------------------------------------------------------------

# fitting the preprocessor and transforming the data
preprocessor.fit(X_train)
X_train_scaled = preprocessor.transform(X_train)
X_test_scaled  = preprocessor.transform(X_test)

# ------------------------------------------------------------------

# Model training and evaluation
def evaluate_model(y_true,y_pred) -> dict[float] : 
    mae = mean_absolute_error(y_true,y_pred)
    mse = mean_squared_error(y_true,y_pred)
    r2  = r2_score(y_true,y_pred) 
    scores = {"mae":mae,"mse":mse,"r2":r2}
    print(scores)
    return scores

# ---------------------------------------------------------------------------   
# Models training and evaluation

lin = LinearRegression()
lin.fit(X_train_scaled,y_train) 
y_pred = lin.predict(X_test_scaled)
evaluate_model(y_test,y_pred)

# ---------------------------------------------------------------------------

forest = RandomForestRegressor()
forest.fit(X_train_scaled,y_train)
y_pred = forest.predict(X_test_scaled)
evaluate_model(y_test,y_pred)

# ---------------------------------------------------------------------------

knn = KNeighborsRegressor()
knn.fit(X_train_scaled,y_train)
y_pred = knn.predict(X_test_scaled)
evaluate_model(y_test,y_pred)

# ---------------------------------------------------------------------------
svm = SVR()
svm.fit(X_train_scaled,y_train)
y_pred = svm.predict(X_test_scaled)
evaluate_model(y_test,y_pred)

# ---------------------------------------------------------------------------
# Bonus : Deep Learning model with Tensorflow

model = keras.Sequential(
    [
        keras.layers.Input(shape=(X_train_scaled.shape[1],))
        ,keras.layers.Dense(128, activation="relu")
        ,keras.layers.Dense(64, activation="relu")
        ,keras.layers.Dense(1)
    ]
)
model.compile(optimizer="adam", loss="mse", metrics=["mae"])
model.fit(X_train_scaled,y_train, epochs=5, batch_size=5, validation_split=0.2)

# --------------------------------
# Hyperparameters tuning with GridSearchCV

grid = {"n_neighbors": range(2,30)}
search = GridSearchCV(KNeighborsRegressor(), grid, verbose=1)
search.fit(X_train_scaled,y_train)
y_pred = search.predict(X_test_scaled)
print(search.best_params_)
evaluate_model(y_test,y_pred)

grid = {"n_estimators": [100,150,300,500]
        #,"max_depth" : [None, 2,5,10]
        #,"min_samples_leaf" : [1,5,10,50]
       }

search = GridSearchCV(RandomForestRegressor(),grid,verbose=1, cv=3)
search.fit(X_train_scaled,y_train)


# %%
y_pred = search.predict(X_test_scaled)
print(search.best_params_)
evaluate_model(y_test,y_pred)

# %% [markdown]
# ### Final Train

# %%
forest = RandomForestRegressor(**search.best_params_)
forest.fit(X_train_scaled,y_train)
y_pred = forest.predict(X_test_scaled)
evaluate_model(y_test,y_pred)

# %% [markdown]
# ## Saving Model

# %%
import pickle 
import os 

# %%
! ls

# %%
# Saving the preprocessor 
model_path = "models"
if not os.path.exists(model_path) : 
    os.mkdir(model_path)
with open(os.path.join(model_path,"preproc.pkl"),"wb") as f:
    pickle.dump(preprocessor,f)

# %%
# Saving the model 
with open(os.path.join(model_path,"model.pkl"),"wb")  as f:
    pickle.dump(forest,f)

# %%
! tree

# %% [markdown]
# ## NB : Loading model

# %%
with open(os.path.join(model_path,"preproc.pkl"),"rb")  as f:
    new_preproc = pickle.load(f)
new_preproc


