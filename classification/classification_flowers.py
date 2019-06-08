# Inspired by https://www.tensorflow.org/alpha/tutorials/keras/feature_columns
# Classification project, will attempt to classify wether cells are cancerous

# Import dependencies
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import pandas as pd

import tensorflow as tf

from tensorflow import feature_column
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split

# Import database, need to add a header, useful for libraries with pandas dataframe and tensorflow
# The data is patient ID (not useful to us), the diagnosis and 30 traits for the cells.
names = ["ID", "Diagnosis",
         "radius", "radius_std", "radius_worst",
         "texture","texture_std","texture_worst",
         "perimeter","perimeter_std","perimeter_worst",
         "area","area_std","area_worst",
         "smoothness","smoothness_std","smoothness_worst",
         "compactness","compactness_std","compactness_worst",
         "concavity","concavity_std","concavity_worst",
         "concave_points","concave_points_std","concave_points_worst",
         "symmetry","symmetry_std","symmetry_worst",
         "fractal_dimension","fractal_dimension_std","fractal_dimension_worst"]
URL = 'http://mlr.cs.umass.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data'
dataframe = pd.read_csv(URL, names=names)
print(dataframe.head())

# split database between train, test and validation
train, test = train_test_split(dataframe, test_size=0.2)
train, val = train_test_split(train, test_size=0.2)
print(len(train), 'train examples')
print(len(val), 'validation examples')
print(len(test), 'test examples')

# prep database to make it useable by model
# A utility method to create a tf.data dataset from a Pandas Dataframe
# All datapoints are numerical, so Ill leave them as they are (some are
# a bit large, and I suspect the model would improve if they were normalized)
# Diagnosis needs to be changed from strings to numerics to be useable by TensorFlow
def df_to_dataset(dataframe, shuffle=True, batch_size=32):
    dataframe = dataframe.copy()
    labels = dataframe.pop('Diagnosis')
    labels = labels.replace(['B', 'M'], [0,1]) 
    dataframe = dataframe.drop(columns='ID')
    ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(dataframe))
    ds = ds.batch(batch_size)
    return ds

# Feature columns will be the input layer and has 30 nodes as it is
feature_columns = []

for header in ["radius", "radius_std", "radius_worst",
               "texture", "texture_std", "texture_worst",
               "perimeter", "perimeter_std", "perimeter_worst",
               "area", "area_std", "area_worst",
               "smoothness", "smoothness_std", "smoothness_worst",
               "compactness", "compactness_std", "compactness_worst",
               "concavity", "concavity_std", "concavity_worst",
               "concave_points", "concave_points_std", "concave_points_worst",
               "symmetry", "symmetry_std", "symmetry_worst",
               "fractal_dimension", "fractal_dimension_std", "fractal_dimension_worst"]:
    feature_columns.append(feature_column.numeric_column(header))

# Choose and make model

feature_layer = tf.keras.layers.DenseFeatures(feature_columns)

batch_size = 16
train_ds = df_to_dataset(train, batch_size=batch_size)
val_ds = df_to_dataset(val, shuffle=False, batch_size=batch_size)
test_ds = df_to_dataset(test, shuffle=False, batch_size=batch_size)

print(type(train_ds))

# Using Dense layers with ReLu activation (essentially AX+b, with A as
# the dense layers and the bias b determined by ReLu), needed multiple Dense layers
# to achieve good accuracy discovered by testing. Finished with sigmoid activation function
# to give an output between 0 and 1
model = tf.keras.Sequential([
    feature_layer,
    layers.Dense(30, activation='relu'),
    layers.Dense(30, activation='relu'),
    layers.Dense(30, activation='relu'),
    layers.Dense(30, activation='relu'),
    layers.Dense(30, activation='relu'),
    layers.Dense(30, activation='relu'),
    layers.Dense(1, activation='sigmoid')
    ])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])


# train model
model.fit(train_ds,
          validation_data=val_ds,
          epochs=10)

# Inspect model to check if it is good
loss, accuracy = model.evaluate(test_ds)
print("Accuracy", accuracy)
