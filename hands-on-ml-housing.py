
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


# Thats on a test-branch