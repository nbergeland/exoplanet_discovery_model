# exoplanet_discovery_model 
README
# Update sklearn to prevent version mismatches
!pip install sklearn --upgrade
# install joblib. This will be used to save our model. 
# Restart your kernel after installing 
!pip install joblib
import pandas as pd
Read the CSV and Perform Basic Data Cleaning
df = pd.read_csv("exoplanet_data.csv")
# Drop the null columns where all values are null
df = df.dropna(axis='columns', how='all')
# Drop the null rows
df = df.dropna()
df.head()
Select your features (columns)
# Set features. This will also be used as your x value
X = df.drop(columns=["koi_disposition"])
Create a Train Test Split
Use koi_disposition for the y values

y_var = df["koi_disposition"]
from sklearn.model_selection import train_test_split 
# Scale your data
X_train, X_test, y_train, y_test = train_test_split(X, y_var, random_state=1, stratify=y_var)
Pre-processing
Scale the data using the MinMaxScaler and perform some feature selection

from sklearn.preprocessing import MinMaxScaler
X_scaler = MinMaxScaler().fit(X_train)
X_train_scaled = X_scaler.transform(X_train)
X_test_scaled = X_scaler.transform(X_test)
Train the Model
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train_scaled, y_train)
print(f"Training Data Score: {model.score(X_train_scaled, y_train)}")
print(f"Testing Data Score: {model.score(X_test_scaled, y_test)}")
Training Data Score: 0.8411214953271028
Testing Data Score: 0.8409610983981693

Hyperparameter Tuning
Use GridSearchCV to tune the model's parameters

from sklearn.model_selection import GridSearchCV
param_grid = {'C': [1, 5, 10],
              'penalty': ["l1", "l2"]}
grid = GridSearchCV(model, param_grid, verbose=3)
# Train the model with GridSearch
grid.fit(X_train_scaled, y_train)
print(grid.best_params_)
print(grid.best_score_)
  {'C': 5, 'penalty': 'l1'}
  0.8800305168796491
Save the Model
# save your model by updating "your_name" with your name
# and "your_model" with your model variable
import joblib
filename = 'your_name.sav'
joblib.dump(model, filename)
['your_name.sav']
​
import tensorflow as tf
from keras.models import Sequential
from keras.utils import to_categorical
from keras.layers import Dense
from sklearn.preprocessing import LabelEncoder
from keras.callbacks import EarlyStopping

label_encoder = LabelEncoder()
label_encoder.fit(y_train)
encoded_y_train = label_encoder.transform(y_train)
encoded_y_test = label_encoder.transform(y_test)
y_train_categorical = to_categorical(encoded_y_train)
y_test_categorical = to_categorical(encoded_y_test)

y_train_categorical

model = Sequential()
model.add(Dense(units=100, activation='relu', input_dim=40))
model.add(Dense(units=100, activation='relu'))
model.add(Dense(units=3, activation='softmax'))

# Compile and fit the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
              
model.summary()
Model: "sequential_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
dense_1 (Dense)              (None, 100)               4100      
_________________________________________________________________
dense_2 (Dense)              (None, 100)               10100     
_________________________________________________________________
dense_3 (Dense)              (None, 3)                 303       
=================================================================
Total params: 14,503
Trainable params: 14,503
Non-trainable params: 0
_________________________________________________________________

# set early stopping as callback
callbacks = [EarlyStopping(monitor='val_loss', patience=2)]
model.fit(
    X_train_scaled,
    y_train_categorical,
    callbacks=callbacks,
    epochs=60,
    shuffle=True,
    verbose=2
)

model_loss, model_accuracy = model.evaluate(
    X_test_scaled, y_test_categorical, verbose=2)
print(
    f"Normal Neural Network - Loss: {model_loss}, Accuracy: {model_accuracy}")

Normal Neural Network - Loss: 0.28015009994092194, Accuracy: 0.8838672637939453 
