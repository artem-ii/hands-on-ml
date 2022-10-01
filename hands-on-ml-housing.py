
# blablabla

# %% Import packages

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


import os, tarfile, urllib


# %% Reading data


DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"


# %%% Fetch


def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    os.makedirs(housing_path, exist_ok=True)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()


# %%% Load


def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)


# %% Explore data


fetch_housing_data()
housing = load_housing_data()
housing.head()


# In[66]:


housing.info()


# In[67]:


# Total bedrooms for 207 block groups are missing


# In[68]:


housing["ocean_proximity"].value_counts()


# In[69]:


housing.describe()


# In[70]:


# get_ipython().run_line_magic('matplotlib', 'inline')
housing.hist(bins=50, figsize=(20, 15))
plt.show()


# ### Random train/test sampling

# %% Splitting, no scikit-learn


def split_train_test(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]


# In[72]:


train_set, test_set = split_train_test(housing, 0.2)


# In[73]:


[len(train_set), len(test_set)]


# In[74]:


# Splitting using hash sum

from zlib import crc32


# In[75]:


def test_set_check(identifier, test_ratio):
    return crc32(np.int64(identifier)) & 0xffffffff < test_ratio * 2**32

def split_train_test_by_id(data, test_ratio, id_column):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio))
    return data.loc[~in_test_set], data.loc[in_test_set]


# In[76]:


'''

It's the simplest way, but it's susceptible to data change
(i.e. addition of new columns in the middle)

'''
housing_with_id = housing.reset_index()
train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "index")


# In[77]:


# A more stable way is to use stable values of instances
housing_with_id["id"] = housing["longitude"] * 1000 + housing["latitude"]
train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "id")


# In[78]:


[len(train_set), len(test_set)]


# %% Splitting scikit-learn
# %%% train_test_split


from sklearn.model_selection import train_test_split


# In[80]:


# Similar to split_train_test() function above, random_state sets seed
train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)


# %%% Stratified sampling

'''

pandas cut() method is used to create bins of median income.
These stratas are important for samples to be representative

'''

housing["income_cat"] = pd.cut(housing["median_income"],
                              bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                              labels=[1, 2, 3, 4, 5])
housing["income_cat"].hist()



from sklearn.model_selection import StratifiedShuffleSplit



split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]


# In[84]:


[len(strat_train_set), len(strat_test_set)]


# In[85]:


strat_test_set["income_cat"].value_counts() / len(strat_test_set)


# In[86]:


for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)


# %%% Data exploration

# In[87]:


# Create a copy of stratified training sample

housing = strat_train_set.copy()


# In[88]:


housing.plot(kind="scatter", x="longitude", y="latitude")


# In[89]:


# Alpha here is proportional to point density as points overlap

housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)


# In[90]:


housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
            s=housing["population"]/100, label="population", figsize=(10,7),
            c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True)
plt.legend()


# In[91]:


corr_matrix = housing.corr()


# In[92]:


corr_matrix["median_house_value"].sort_values(ascending=False)


# In[93]:


from pandas.plotting import scatter_matrix


# In[94]:


attributes = ["median_house_value", "median_income", "total_rooms",
             "housing_median_age"]
scatter_matrix(housing[attributes], figsize=(12, 8))


# In[95]:


# Zoom into the median income plot

housing.plot(kind="scatter", x="median_income", y="median_house_value",
            alpha=0.1)


# %% Attribute combinations

housing["rooms_per_household"] = housing["total_rooms"]/housing["households"]
housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"]
housing["population_per_household"] = housing["population"]/housing["households"]


# %%% New correlation matrix

corr_matrix = housing.corr()
corr_matrix["median_house_value"].sort_values(ascending=False)


# %% Prepare Data

housing = strat_train_set.drop("median_house_value", axis=1)
housing_labels = strat_train_set["median_house_value"].copy()



# %%% Data Cleaning
# %%%% No scikit-learn
# %%%%% Drop rows with missing values

housing.dropna(subset=["total_bedrooms"])

# %%%%% Drop problematic column

housing.drop("total_bedrooms", axis=1)

# %%%%% Fill with median values

median = housing["total_bedrooms"].median()
housing["total_bedrooms"].fillna(median, inplace=True)

# %%%% Scikit-learn way
# %%%%% Import imputer
from sklearn.impute import SimpleImputer

# %%%%% Imputer use

imputer = SimpleImputer(strategy="median")
# Create copy of dataset mith only numerical values
housing_num = housing.drop("ocean_proximity", axis=1)
imputer.fit(housing_num)

imputer.statistics_

housing_num.median().values

X = imputer.transform(housing_num)
housing_tr = pd.DataFrame(X, columns=housing_num.columns,
                          index=housing_num.index)

# There are estimator, transformer and predictor objects in scikit-learn

# %%%% Text and categorical values

housing_cat = housing[["ocean_proximity"]]
housing_cat.head(10)

# %%%%% Ordinal encoder
# Ordinal encoder creates an R-like 'factor' out of categorical variable

from sklearn.preprocessing import OrdinalEncoder
ordinal_encoder = OrdinalEncoder()
housing_cat_encoded = ordinal_encoder.fit_transform(housing_cat)
housing_cat_encoded[:10]
ordinal_encoder.categories_

# %%%%% One-hot encoder
'''

This encoder transforms categorical variable into a design matrix-like
structure, i.e. one column per category, marking rows of particular category
with ones and the rest of rows with zeros 

The benefit of a one-hot encoder for this application is that it doesn't
imply that any of the categories can be compared mathematically to others.

'''

from sklearn.preprocessing import OneHotEncoder
cat_encoder = OneHotEncoder()
housing_cat_1hot = cat_encoder.fit_transform(housing_cat)
housing_cat_1hot

'''

Pay attention to sparse and dense matrices. Sparse matrix only stores
positions of non-zero elements

'''
# A way to convert sparse to dense matrix
housing_cat_1hot.toarray()

cat_encoder.categories_

# %%%% Feature scaling

'''

To common ways: min-max scaling (normalization) - bound to [0,1] range
and standardization (based on mean) - protects from outliers

'''

# %%% Custom transformers

from sklearn.base import BaseEstimator, TransformerMixin

rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6

class CombinedAttributeAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room=True):
        self.add_bedrooms_per_room = add_bedrooms_per_room
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
        population_per_household = X[:, population_ix] / X[:, households_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household,
                         bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]
        
attr_adder = CombinedAttributeAdder(add_bedrooms_per_room=False)
housing_extra_attribs = attr_adder.transform(housing.values)

# %%% Transformation pipelines

# %%%% Import Pipeline
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# %%%% Numeric pipeline

num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),
    ('attribs_adder', CombinedAttributeAdder()),
    ('std_scaler', StandardScaler())
    ])

housing_num_tr = num_pipeline.fit_transform(housing_num)

# %%%% Full pipeline
'''

Here categorical variable will also be included with respective behavior.

'''

from sklearn.compose import ColumnTransformer

num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]

full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_attribs),
    ("cat", OneHotEncoder(), cat_attribs)
    ])

housing_prepared = full_pipeline.fit_transform(housing)


# %% Model Training

# %%% LinearRegression

from sklearn.linear_model import LinearRegression


# %%%% Train

lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)


#%%%% Evaluate

some_data = housing.iloc[:5]
some_labels = housing_labels.iloc[:5]
some_data_prepared = full_pipeline.transform(some_data)
print("Predictions: ", lin_reg.predict(some_data_prepared))
print("Labels: ", list(some_labels))

'''

A bit weird that I got values different from the book:

Predictions:  [ 85657.90192014 305492.60737488 152056.46122456 186095.70946094
 244550.67966089]
Labels:  [72100.0, 279600.0, 82700.0, 112500.0, 238300.0]

'''
from sklearn.metrics import mean_squared_error

housing_predictions = lin_reg.predict(housing_prepared)
lin_mse = mean_squared_error(housing_labels, housing_predictions)
lin_rmse = np.sqrt(lin_mse)

lin_rmse

'''

Prediction error is still very close to the book:
    
Out[59]: 68627.87390018745

Error too big on training data — Underfitting
Solutions:
    - More powerful model <<<
    - Better features
    - Reduce model condtraints

'''

# %%% DecisionTreeRegressor

from sklearn.tree import DecisionTreeRegressor


# %%%% Train

tree_reg = DecisionTreeRegressor()
tree_reg.fit(housing_prepared, housing_labels)


# %%%% Evaluate

# %%%%% MSE

housing_predictions = tree_reg.predict(housing_prepared)
tree_mse = mean_squared_error(housing_labels, housing_predictions)
tree_rmse = np.sqrt(tree_mse)

tree_rmse

'''
Now prediction error equals zero, which is an example of overfitting:
Out[63]: 0.0

But we can't be 100% sure (maybe the model is perfect). At the same time
we can't touch the test set.

The solution here is to try cross-validation (mincing the training set).

'''

# %%%%% Cross-validation
# Using scikit-learn's K-fold cross-validation

from sklearn.model_selection import cross_val_score

scores = cross_val_score(tree_reg, housing_prepared, housing_labels,
                         scoring="neg_mean_squared_error", cv=10)
tree_rmse_scores = np.sqrt(-scores)

'''
Scikit-learn cross-validation expect a utility function (greater better)
rather than a cost function (lower better). That is why Negative MSE
and a "-" before "scores"

'''


def display_scores(scores):
    print("Scores: ", scores)
    print("Mean: ", scores.mean())
    print("Standard deviation: ", scores.std())
    
    
display_scores(tree_rmse_scores)

lin_scores = cross_val_score(lin_reg, housing_prepared, housing_labels,
                             scoring="neg_mean_squared_error", cv=10)
lin_rmse_scores = np.sqrt(-lin_scores)
display_scores(lin_rmse_scores)

'''

Now we can see that Decision tree actually performs worse
than a linear regression:

    
Decision Tree:
    
Scores:  [73050.00666094 70862.41020418 67162.19680021 71974.22002155
 70937.30426538 77950.42455228 69945.11545533 73668.8243465
 68425.01862088 72419.28649365]
Mean:  71639.48074208843
Standard deviation:  2846.095226420542


Linear Regression:

Scores:  [71762.76364394 64114.99166359 67771.17124356 68635.19072082
 66846.14089488 72528.03725385 73997.08050233 68802.33629334
 66443.28836884 70139.79923956]
Mean:  69104.07998247063
Standard deviation:  2880.328209818068


This proves bad overfitting by the Decision Tree model.
Next idea is to try an ensemble learning approach.

'''

# %%% RandomForestRegressor

from sklearn.ensemble import RandomForestRegressor

'''
It works by training many Decision Tree regressors and averaging results.

'''


# %%%% Train

forest_reg = RandomForestRegressor()
forest_reg.fit(housing_prepared, housing_labels)

# %%%% Cross-validation

forest_scores = cross_val_score(forest_reg, housing_prepared, housing_labels,
                                scoring="neg_mean_squared_error", cv=10)
forest_rmse_scores = np.sqrt(-forest_scores)
display_scores(forest_rmse_scores)

'''

This is somewhat better:

Scores:  [51341.78863615 48811.86798038 46737.89664285 52009.23908427
 47568.14348811 51786.79228231 52541.34890257 49708.64476274
 48970.6817542  53812.21340615]
Mean:  50328.861693974584
Standard deviation:  2192.0601335620045

However, the prediction errors are lower on the training set, that those in
cross-validation;

'''

# %%%% MSE

housing_predictions = forest_reg.predict(housing_prepared)
forest_mse = mean_squared_error(housing_labels, housing_predictions)
forest_rmse = np.sqrt(forest_mse)
forest_rmse

'''

MSE on a training set gives:
Out[69]: 18864.37419588292

Thus, still overfitting is going on.

Solutions:
    - Simplify model
    - Constrain model (regularize)
    - Get a lot more training data
    
But trying only two models is not a good idea. Better try much more and
shortlist a few to further tweak them.

'''

# %% Save every model

'''

There's an optimized "joblib" library based on pickle to save the models

''' 

import joblib

def save_model(model, data, cv_scores):
    model_dump = {
        'model': model,
        'cv-scores': cv_scores,
        'predictions': model.predict(data)
        }
    os.makedirs("processed_data", exist_ok=True)
    dump_path = os.path.join("processed_data", str(model) + ".pkl")
    joblib.dump(model_dump, dump_path)
    print("Saved " + str(dump_path))
    
save_model(lin_reg, housing_prepared, lin_rmse_scores)
save_model(tree_reg, housing_prepared, tree_rmse_scores)
save_model(forest_reg, housing_prepared, forest_rmse_scores)

'''
Saved processed_data/LinearRegression().pkl
Saved processed_data/DecisionTreeRegressor().pkl
Saved processed_data/RandomForestRegressor().pkl

'''

# %% Fine-tune model

# %%% Grid search

'''

Grid Search allows to select best combinations
of hypermarameters automatically.

'''

from sklearn.model_selection import GridSearchCV

param_grid = [
    {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]}
    ]

forest_reg = RandomForestRegressor()

grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
                           scoring="neg_mean_squared_error",
                           return_train_score=True)
grid_search.fit(housing_prepared, housing_labels)

grid_search.best_params_

'''

Out[73]: {'max_features': 8, 'n_estimators': 30}

both max_features and n_estimators are the highest. So may be good to increase

'''

grid_search.best_estimator_

cvres = grid_search.cv_results_

for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)
    
'''

63656.41825881063 {'max_features': 2, 'n_estimators': 3}
54794.56777899004 {'max_features': 2, 'n_estimators': 10}
52879.85414144393 {'max_features': 2, 'n_estimators': 30}
60848.98124399648 {'max_features': 4, 'n_estimators': 3}
52697.542516577436 {'max_features': 4, 'n_estimators': 10}
50386.57422609593 {'max_features': 4, 'n_estimators': 30}
59209.88791866784 {'max_features': 6, 'n_estimators': 3}
52338.10226104821 {'max_features': 6, 'n_estimators': 10}
50123.2811221562 {'max_features': 6, 'n_estimators': 30}
58508.49558209069 {'max_features': 8, 'n_estimators': 3}
52706.53727165742 {'max_features': 8, 'n_estimators': 10}
>>> 50074.376025683116 {'max_features': 8, 'n_estimators': 30} <<<
62905.18300525379 {'bootstrap': False, 'max_features': 2, 'n_estimators': 3}
54182.008605271876 {'bootstrap': False, 'max_features': 2, 'n_estimators': 10}
59828.744141675365 {'bootstrap': False, 'max_features': 3, 'n_estimators': 3}
52811.72591554733 {'bootstrap': False, 'max_features': 3, 'n_estimators': 10}
59259.320532715625 {'bootstrap': False, 'max_features': 4, 'n_estimators': 3}
51474.61380983535 {'bootstrap': False, 'max_features': 4, 'n_estimators': 10}

Also, custom hyperparameters may be added. Like exclusion/inclusion
of particular features.

'''

# %%% Randomized Search

'''

May be used with the help of RandomizedSearchCV().
Is better for big hyperparameter spaces because doesn't evaluate all the
combinations and thus requires less computing power.

'''

# %%% Ensemble models

'''
May improve results.

'''

# %% Best models
# Now analysis of the best models and their errors is needed

# Which features are the most important for making predictions?
feature_importances = grid_search.best_estimator_.feature_importances_
feature_importances

extra_attribs = ["rooms_per_hhold", "pop_per_hhold", "bedrooms_per_room"]
cat_encoder = full_pipeline.named_transformers_["cat"]
cat_one_hot_attribs = list(cat_encoder.categories_[0])
attributes = num_attribs + extra_attribs + cat_one_hot_attribs
sorted(zip(feature_importances, attributes), reverse=True)

'''

[(0.36912944666255143, 'median_income'),
 (0.16060236730134156, 'INLAND'),
 (0.11153172077859291, 'pop_per_hhold'),
 (0.07035800652141727, 'longitude'),
 (0.06568491169735001, 'latitude'),
 (0.053572026999467816, 'rooms_per_hhold'),
 (0.049566482835639976, 'bedrooms_per_room'),
 (0.04441521791263773, 'housing_median_age'),
 (0.01565476145109635, 'total_rooms'),
 (0.014537866785218326, 'total_bedrooms'),
 (0.014524373472261072, 'population'),
 (0.013795810721112177, 'households'),
 (0.011156959918248952, '<1H OCEAN'),
 (0.003752039354409678, 'NEAR OCEAN'),
 (0.0016475249448524645, 'NEAR BAY'),
 (7.048264380228585e-05, 'ISLAND')]

As some features are less relevant than others, we may remove some of them.

'''

# %% Evaluate on Test set

# Don't fit the test set!!

final_model = grid_search.best_estimator_

X_test = strat_test_set.drop("median_house_value", axis=1)
y_test = strat_test_set["median_house_value"].copy()

X_test_prepared = full_pipeline.transform(X_test)

final_predictions = final_model.predict(X_test_prepared)

final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)

# %%% How precise?

# Computing a 95% confidence interval for generalization error

from scipy import stats
confidence = 0.95
squared_errors = (final_predictions - y_test) ** 2
np.sqrt(stats.t.interval(confidence, len(squared_errors) - 1,
                         loc=squared_errors.mean(),
                         scale=stats.sem(squared_errors))
        )

'''

Out[82]: array([46039.75485988, 50017.30373821])

'''