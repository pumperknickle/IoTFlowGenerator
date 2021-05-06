import sys
import numpy as np
import pickle
from keras.models import Model
from keras.layers import Input, Dense
from keras.optimizers import Adam
from keras.utils.vis_utils import plot_model
from sklearn.model_selection import train_test_split
from keras.layers import LSTM, Concatenate
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from utilities import multiply_durations, concat_all, get_max_packet_size, extract_packet_sizes, extract_packet_size_ranges, token_frequency_features_and_labels, packet_size_features, generate_traffic_rate_features, splitAllFeatures, extract_durations, normalize_packet_sizes
import warnings
from sklearn.metrics import confusion_matrix, matthews_corrcoef, f1_score
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import os
import random
import csv
from statistics import multimode

def durationsToTimestamp(all_durations, max_duration=1.0):
    all_timestamps = []
    for durations in all_durations:
        timestamps = []
        float_durations = [float(x) for x in durations]
        for i in range(len(float_durations)):
            timestamps.append(sum(float_durations[0:i+1]) * max_duration)
        all_timestamps.append(timestamps)
    return all_timestamps

def flatten(features):
    flattened = []
    for feature in features:
        for f in feature:
            flattened.append(f)
    return flattened

sys.setrecursionlimit(1000000)

warnings.filterwarnings('ignore')

# fix random seed for reproducibility
np.random.seed(7)

featuresFilePath = sys.argv[1]

with open(featuresFilePath, mode='rb') as featuresFile:
    raw_features = pickle.load(featuresFile)

raw_features = splitAllFeatures(raw_features)
all_keys = raw_features.keys()
features = []
labels = []
id_to_device = dict()
accuracies = dict()
f1scores = dict()
f1normscores = dict()
mccscores = dict()


for key in all_keys:
    value = raw_features[key]
    real = generate_traffic_rate_features(concat_all(extract_packet_sizes(value), durationsToTimestamp(extract_durations(value))))
    fake = []
    for a in all_keys:
        if a != key:
            fakeValue = raw_features[a]
            fake += generate_traffic_rate_features(concat_all(extract_packet_sizes(fakeValue), durationsToTimestamp(extract_durations(fakeValue))))
    test_split_factor = 0.33
    total_fakes_needed = len(fake) * test_split_factor
    if len(real) * 2 < 100:
        extra_fake_features, fake = train_test_split(fake, test_size=100 / len(fake), random_state=42)
        if len(fake) * test_split_factor < total_fakes_needed:
            extra_fake_features = random.sample(extra_fake_features, int(total_fakes_needed - (len(fake) * test_split_factor)))
        else:
            extra_fake_features = []
    else:
        if len(real) * 2 < len(fake):
            extra_fake_features, fake = train_test_split(fake, test_size=len(real) * 2 / len(fake), random_state=42)
            if len(fake) * test_split_factor < total_fakes_needed:
                extra_fake_features = random.sample(extra_fake_features, int(total_fakes_needed - (len(fake) * test_split_factor)))
            else:
                extra_fake_features = []
        else:
            extra_fake_features = []

    labels = np.array([0] * (len(real)) + ([1] * len(fake)))
    features = np.array(real + fake)
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=test_split_factor, random_state=42)
    X_test = np.array(list(X_test) + extra_fake_features)
    y_test = np.array(list(y_test) + ([1] * len(extra_fake_features)))
    X_train_final = []
    y_train_final = []
    for i in range(len(X_train)):
        X_train_final += X_train[i]
        y_train_final += [y_train[i]] * len(X_train[i])

    neigh = KNeighborsClassifier(n_neighbors=5)
    neigh.fit(np.array(X_train_final), np.array(y_train_final))

    predicted = []
    correct = 0
    wrong = 0
    positives = 0
    negatives = 0

    for i in range(len(X_test)):
        test_feat = X_test[i]
        test_label = y_test[i]
        prediction = neigh.predict(test_feat)
        mm = multimode(prediction)
        if 1 in mm:
            predicted.append(1)
            if test_label == 1:
                correct += 1
                negatives += 1
            else:
                wrong += 1
                positives += 1
        else:
            predicted.append(0)
            if test_label == 1:
                wrong += 1
                negatives += 1
            else:
                correct += 1
                positives += 1

    accuracy = correct / (correct + wrong)
    print(key)
    print("Accuracy of model is ")
    print(accuracy)
    accuracies[key] = accuracy
    results = confusion_matrix(y_test, predicted)
    print(results)
    print("f1 score")
    f1 = f1_score(y_test, predicted, pos_label=0, average='binary')
    print(f1)
    f1scores[key] = f1
    q = positives / (positives + negatives)
    f1_coin = (2 * q) / (q + 1)
    f1_norm = (f1 - f1_coin) / (1 - f1_coin)
    print("f1 norm")
    print(f1_norm)
    f1normscores[key] = f1_norm

    mcc = matthews_corrcoef(y_test, predicted)
    print("mcc")
    print(mcc)
    mccscores[key] = mcc

with open("trafficrateaccuraciesexp0.pkl", mode='wb') as featureOutputFile:
    pickle.dump(accuracies, featureOutputFile)

with open("trafficratef1experiment0.pkl", mode='wb') as featureOutputFile:
    pickle.dump(f1scores, featureOutputFile)

with open("trafficratef1normexperiment0.pkl", mode='wb') as featureOutputFile:
    pickle.dump(f1normscores, featureOutputFile)

with open("trafficratemccscoresexp1.pkl", mode='wb') as featureOutputFile:
    pickle.dump(mccscores, featureOutputFile)