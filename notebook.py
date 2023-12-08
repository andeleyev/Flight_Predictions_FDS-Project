# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # Final Project: **Regression Analysis of German Air Fares**
# by **Andre Datchev**, **Hannah Kiel**, **Hannes Pohnke**, **Nikolas Jochens**

# ## **1) Opening Remarks**
#
# We don't know exactly what kind of price configurations were chosen while scraping the dataset. We are assuming that each entry is the cheapest possible configuration for the corresponding flight. So no additional baggage, no extra leg room, no business class and no other extras that would influence the price.

# ## **2) Setup and Dataset Cleanup**

# ### **2.1) Imports**

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import datetime
import random

# ### **2.2) Dataset**
# We will now import our dataset. It can be found under:  [*https://www.kaggle.com/datasets/darjand/domestic-german-air-fares*](https://www.kaggle.com/datasets/darjand/domestic-german-air-fares)<br>
# We renamed the csv file to *german_air_fares.csv*.

data = pd.read_csv("german_air_fares.csv", sep=";")
data.head()
data.head()
data.info()
data.describe() 

# ### **2.3) Missing Values**

print(data.isnull().sum())
data = data.dropna() # we can just drop this little number of missing values

# ### **2.4) Ambiguous Values & Cleanup**
#
# Problem: "1 Stopp" and "1 stop" is the same, but in two different languages. We have the same problem with "Mehrere Fluglinien" and "Multiple Airlines".

# +
# set initial data types
data = data.astype('string')

# rename price column
data = data.rename(columns={'price (€)': 'price'})

# fix price column format errors
data.price = data.price.str.replace(',', '')
data.price = data.price.str.replace('.00', '')

# rename entries in airlines column
data.airline = data.airline.str.replace('Mehrere Fluglinien', 'Multiple Airlines')

# rename entries in stops column
data.stops = data.stops.str.replace('direct', '0')
data.stops = data.stops.str.replace('(1 Stopp)', '1')
data.stops = data.stops.str.replace('(1 stop)', '1')
data.stops = data.stops.str.replace('(2 Stopps)', '2')

# rename entries in departure_date_distance
data.departure_date_distance = data.departure_date_distance.str.replace('2 weeks', '2 week')
data.departure_date_distance = data.departure_date_distance.str.replace('2 week', '2 weeks')
data.departure_date_distance = data.departure_date_distance.str.replace('3 months', '3 month')
data.departure_date_distance = data.departure_date_distance.str.replace('3 month', '3 months')

# rename entries in departure_time column
data.departure_time = data.departure_time.str.replace(' Uhr', '')

# update data types
#data.price = data.price.str.replace('', '0')
data = data.astype({'price': 'int32'})

# delete erroneous row
data = data.drop(data[data.price == data.price.max()].index)

data = data.astype({'stops': 'int32'})
data = data.astype({'scrape_date': 'datetime64[ns]'})
data = data.astype({'departure_date': 'datetime64[ns]'})
data = data.astype({'departure_time': 'datetime64[ns]'})


# -

# ### **2.5) Feature Augmentation**
# //features adden (mean,...?)

# +
#stuff
# -

# ### **2.6) Encoding of categorical Features & Standardization of numerical Features**

# +
features = data[['airline', 'stops', 'departure_date', 'departure_time', 'departure_date_distance']]
label = data['price']

# One-hot encode categorical features and scale numerical features
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), ['stops']),
        ('cat', OneHotEncoder(), ['airline', 'departure_date_distance'])
    ])

X = preprocessor.fit_transform(features)
y = label.values
# -

# ### **2.7) Train/Test Split**

#could stratify departure date, but not important imo because there are no dates with a really low number of entries 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#

data.dtypes

data.head()

# ## **3) Analysis of Dataset**

# +
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(20, 10))

data.airline.value_counts().plot(ax=axes[0,0], kind="bar", xlabel="")
_ = axes[0,0].set_title('Airlines')

data.departure_city.value_counts().plot(ax=axes[0,1], kind="bar", xlabel="")
_ = axes[0,1].set_title('Departure City')

data.arrival_city.value_counts().plot(ax=axes[0,2], kind="bar", xlabel="")
_ = axes[0,2].set_title('Arrival City')

data.stops.value_counts().plot(ax=axes[1,0], kind="bar", xlabel="")
_ = axes[1,0].set_title('Number of Stops')

data.price.plot(ax=axes[1,1], kind="kde", xlabel="", xlim=(0,1150))
_ = axes[1,1].set_title('Price Density')

data.departure_date.value_counts().sort_index().plot(ax=axes[1,2], kind="bar", xlabel="")
_ = axes[1,2].set_title('Departure Date')

plt.subplots_adjust(hspace=1)


# -

# ### **3.3) Median & Mean Price by Weekday, Month and Departure Date Distance**

# +
def print_price_distributions(dataset):

    # group by weekday & month & departure date distance
    grouped_weekday = dataset.groupby(dataset.departure_date.dt.day_name())
    grouped_month = dataset.groupby(dataset.departure_date.dt.month_name())
    grouped_departure_distance = dataset.groupby(dataset.departure_date_distance)

    # calculate mean and median prices
    weekday_mean_price = grouped_weekday.price.mean().reindex(["Monday","Tuesday","Wednesday","Thursday","Friday", "Saturday", "Sunday"])
    weekday_median_price = grouped_weekday.price.median().reindex(["Monday","Tuesday","Wednesday","Thursday","Friday", "Saturday", "Sunday"])
    weekday_min_price = grouped_weekday.price.min().reindex(["Monday","Tuesday","Wednesday","Thursday","Friday", "Saturday", "Sunday"])

    month_mean_price = grouped_month.price.mean().reindex(["January","April","October","November","December"])
    month_median_price = grouped_month.price.median().reindex(["January","April","October","November","December"])
    month_min_price = grouped_month.price.min().reindex(["January","April","October","November","December"])

    departure_distance_mean_price = grouped_departure_distance.price.mean().reindex(["1 week","2 weeks","1 month","6 weeks","3 months", "6 months"])
    departure_distance_median_price = grouped_departure_distance.price.median().reindex(["1 week","2 weeks","1 month","6 weeks","3 months", "6 months"])
    departure_distance_min_price = grouped_departure_distance.price.min().reindex(["1 week","2 weeks","1 month","6 weeks","3 months", "6 months"])

    # plot mean and median prices
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(20, 6))
    plt.subplots_adjust(bottom=0.22)

    _ = weekday_mean_price.plot(ax=axes[0], xlabel="", kind="bar", color="darkblue")
    _ = weekday_median_price.plot(ax=axes[0], xlabel="", kind="bar")
    _ = weekday_min_price.plot(ax=axes[0], xlabel="", kind="bar", color="lightblue")
    _ = axes[0].set_title('Weekday')

    _ = month_mean_price.plot(ax=axes[1], xlabel="", kind="bar", color="darkblue")
    _ = month_median_price.plot(ax=axes[1], xlabel="", kind="bar")
    _ = month_min_price.plot(ax=axes[1], xlabel="", kind="bar", color="lightblue")
    _ = axes[1].set_title('Month')

    _ = departure_distance_mean_price.plot(ax=axes[2], xlabel="", kind="bar", color="darkblue")
    _ = departure_distance_median_price.plot(ax=axes[2], xlabel="", kind="bar")
    _ = departure_distance_min_price.plot(ax=axes[2], xlabel="", kind="bar", color="lightblue")
    _ = axes[2].set_title('Departure Date Distance')

    _ = fig.suptitle('Price Comparison', fontsize=22)
    _ = fig.legend(labels=["Mean Price", "Median Price", "Minimum Price"], loc="lower center", ncol=3, frameon=False, fontsize=12)

print_price_distributions(data)
# -

# ### **3.4) Mean price per airport**



# ### **3.5) Median & Mean Price per Stop**



# ## **4) Train & Test Subsets**

# ### **4.2) Add mean, median, min and max as new columns**

# +
# prepare for subset creation
new_data = data.drop(['scrape_date', 'arrival_time', 'departure_time', 'airline', 'stops'], axis=1)

# group by date
grouped_date = data.groupby(data.departure_date)

# calculate mean, median, maximum and minimum price for each day
day_mean_price = grouped_date.price.mean().rename('mean_price')
day_median_price = grouped_date.price.median().rename('median_price')
day_min_price = grouped_date.price.min().rename('min_price')
day_max_price = grouped_date.price.max().rename('max_price')

# add mean, median, maximum and minimum price columns to data
new_data = new_data.merge(day_mean_price, on='departure_date')
new_data = new_data.merge(day_median_price, on='departure_date')
new_data = new_data.merge(day_min_price, on='departure_date')
new_data = new_data.merge(day_max_price, on='departure_date')
# -

# ### **4.3) Perform train/test split while keeping single days together**

# +
# sort by departure date
new_data = new_data.sort_values(by='departure_date')

# Perform split while keeping days together
train, test = train_test_split(new_data, test_size=0.2, stratify=new_data['departure_date'])
# -

print_price_distributions(train)
print_price_distributions(test)

# +
# reorder columns
departure_date_column = new_data.pop('departure_date') 
new_data.insert(0, 'departure_date', departure_date_column)
new_data.sort_index(inplace=True)
new_data.sort_index(axis=1)

# get unique values
unique_days = new_data.departure_date.unique()

# randomly get 10 of the 40 days contained in new_data
test_split_temp = pd.DataFrame(unique_days, columns=["departure_date"]).sample(n=10, random_state=200)

test_split = pd.merge(test_split_temp, new_data, how ='inner', on =['departure_date', 'departure_date'])
test_split.sort_index(inplace=True)
test_split.sort_index(axis=1)


train_split = pd.DataFrame(index=new_data.index, columns=new_data.columns)

# Iterate through columns and rows to find differences
for row in new_data.columns:
    train_split[row] = ~new_data[row].equals(test_split[row])

print(len(new_data))
print(len(test_split))
print(len(train_split))

# divide by months
# divide by weekdays
# divide by departure_date_distance

#new_data.head()
# -

# ## **5) Training of Regression Models**

X = train.iloc[:, 0].values.reshape(-1, 1)  # values converts it into a numpy array
Y = train.iloc[:, 1].values.reshape(-1, 1)  # -1 means that calculate the dimension of rows, but have 1 column
print(X)
print(Y)
#linear_regressor = LinearRegression()  # create object for the class
#linear_regressor.fit(X, Y)  # perform linear regression
#Y_pred = linear_regressor.predict(X)  # make predictions

# ### **5.1) Linear Regression**

# +
def hyphotesis(x, theta):
    
    return np.dot(x, theta)

def error_function1(x, y, theta):
    h = hyphotesis(x, theta)
    
    return 1/(2 * len(y)) * np.sum(np.square(h - y))

def error_function2():
    
    return

def gradient_descent():
    
    return 

def linear_regression():
    
    return


# -

# ### **5.2) Polynomial Regression**

# ## **6) Training of Neural Network**

# +
#pip install tensorflow

# +
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, InputLayer
from sklearn.compose import ColumnTransformer

# Load the dataset
data = pd.read_csv("german_air_fares.csv", sep=";")

# Apply the preprocessing steps as you described
data = data.astype('string')
data = data.rename(columns={'price (€)': 'price'})
data.price = data.price.str.replace(',', '').str.replace('.00', '')
data.airline = data.airline.str.replace('Mehrere Fluglinien', 'Multiple Airlines')
data.stops = data.stops.str.replace('direct', '0').str.replace('(1 Stopp)', '1').str.replace('(1 stop)', '1').str.replace('(2 Stopps)', '2')
data.departure_date_distance = data.departure_date_distance.str.replace('2 weeks', '2 week').str.replace('2 week', '2 weeks').str.replace('3 months', '3 month').str.replace('3 month', '3 months')
data.departure_time = data.departure_time.str.replace(' Uhr', '')
data = data.astype({'price': 'int32'})
data = data.drop(data[data.price == data.price.max()].index)
data = data.astype({'stops': 'int32'})
data = data.astype({'scrape_date': 'datetime64[ns]', 'departure_date': 'datetime64[ns]', 'departure_time': 'datetime64[ns]'})

# Select features and label
features = data[['airline', 'stops', 'departure_date', 'departure_time', 'departure_date_distance']]
label = data['price']

# One-hot encode categorical features and scale numerical features
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), ['stops']),
        ('cat', OneHotEncoder(), ['airline', 'departure_date_distance'])
    ])

X = preprocessor.fit_transform(features)
y = label.values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# +

# Create a neural network model
model = Sequential([
    InputLayer(input_shape=(X_train.shape[1],)),
    Dense(64, activation='relu'),
    Dense(64, activation='relu'),
    Dense(1)
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32)

# Evaluate the model
loss = model.evaluate(X_test, y_test)
print(f"Test loss: {loss}")


# +
from sklearn.metrics import mean_absolute_error, r2_score

import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score


predictions = model.predict(X_test)

for i in range(100):  # Anzeigen der ersten 10 Vorhersagen
    print(f"Vorhergesagter Preis: {predictions[i][0]}, Tatsächlicher Preis: {y_test[i]}")

# Ihr vorheriger Code, um das Modell zu trainieren und Vorhersagen zu machen...

# Berechnen der Metriken
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

# Ausgabe der Metriken
print(f"Mittlerer absoluter Fehler (MAE): {mae:.2f}")
print(f"R²-Wert: {r2:.2f}")

# Visualisierung der Vorhersagen im Vergleich zu den tatsächlichen Werten
plt.figure(figsize=(10, 6))
plt.scatter(y_test, predictions, alpha=0.5)
plt.xlabel('Tatsächliche Preise')
plt.ylabel('Vorhergesagte Preise')
plt.title('Vorhergesagte Preise vs Tatsächliche Preise')
max_price = max(y_test.max(), predictions.max())
min_price = min(y_test.min(), predictions.min())
plt.plot([min_price, 1000], [min_price, 1000], 'r--')  # Rote gestrichelte Linie

plt.show()
# -

# ## **7) Comparison of Models**

# ## **8) Final Thoughts**
