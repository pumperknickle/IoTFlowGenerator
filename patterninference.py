import sys
import numpy as np
import pickle
from keras.models import Sequential
from keras.layers import Input, Dense
from keras.utils.vis_utils import plot_model
from sklearn.model_selection import train_test_split
from utilities import dbclustermin, ngrams, extractFeatures, extractSignatures, extract_packet_sizes, splitAllFeatures
import warnings
from sklearn.metrics import confusion_matrix, f1_score, matthews_corrcoef
import random

# distance metric used by dbscan
distance_threshold = 5.0
# total ngrams divided by cluster threshold is equal to the min_samples needed to form a cluster in dbscan
min_cluster = 4
min_sig_size = 2
max_sig_size = 5

sys.setrecursionlimit(1000000)

warnings.filterwarnings('ignore')

# fix random seed for reproducibility
np.random.seed(7)

featuresFilePath = sys.argv[1]

with open(featuresFilePath, mode='rb') as featuresFile:
    raw_features = pickle.load(featuresFile)

device_to_packet_sizes = dict()
raw_features = splitAllFeatures(raw_features)

all_keys = []

all_fake_features = dict()
all_real_features = dict()

eventinferenceresults = dict()
eventinferencef1 = dict()
eventinferencef1norm = dict()
mccscores = dict()

for key, value in raw_features.items():
    all_keys.append(key)
    device_to_packet_sizes[key] = extract_packet_sizes(value)

def baseline_model():
  model = Sequential()
  model.add(Dense(300, activation='relu'))
  model.add(Dense(300, activation='relu'))
  model.add(Dense(2, activation='softmax'))
  model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
  return model

for key in all_keys:
    real = device_to_packet_sizes[key]
    fake = []
    for a in all_keys:
        if a == key:
            continue
        fake += device_to_packet_sizes[a]
    if (len(fake) == 0 or len(real) == 0):
        continue
    real_features = [None] * len(real)
    fake_features = [None] * len(fake)
    for i in range(min_sig_size, max_sig_size + 1):
        allngrams = []
        for feature in real:
            ngramVector = ngrams(i, feature)
            for ngram in ngramVector:
                allngrams.append(ngram)
        cluster = dbclustermin(allngrams, distance_threshold, min_cluster)
        signatures = extractSignatures(cluster, i)
        for n in range(len(real)):
            feature = real[n]
            extractedNgrams = ngrams(i, feature)
            newFeatures = extractFeatures(extractedNgrams, signatures)
            if real_features[n] == None:
                real_features[n] = newFeatures
            else:
                real_features[n] = real_features[n] + newFeatures
        for n in range(len(fake)):
            feature = fake[n]
            extractedNgrams = ngrams(i, feature)
            newFeatures = extractFeatures(extractedNgrams, signatures)
            if fake_features[n] == None:
                fake_features[n] = newFeatures
            else:
                fake_features[n] = fake_features[n] + newFeatures
    test_split_factor = 0.33
    total_fakes_needed = len(fake_features) * test_split_factor
    if len(real_features) * 2 < 100:
        extra_fake_features, fake_features = train_test_split(fake_features, test_size=100/len(fake_features), random_state=42)
        if len(fake_features) * test_split_factor < total_fakes_needed:
            extra_fake_features = random.sample(extra_fake_features, int(total_fakes_needed - (len(fake_features) * test_split_factor)))
        else:
            extra_fake_features = []
    else:
        if len(real_features) * 2 < len(fake_features):
            extra_fake_features, fake_features = train_test_split(fake_features, test_size= len(real_features) * 2 / len(fake_features), random_state=42)
            if len(fake_features) * test_split_factor < total_fakes_needed:
                extra_fake_features = random.sample(extra_fake_features, int(total_fakes_needed - (len(fake_features) * test_split_factor)))
            else:
                extra_fake_features = []
        else:
            extra_fake_features = []
    labels = np.array([0] * (len(real_features)) + ([1] * len(fake_features)))
    features = np.array(list(real_features) + list(fake_features))
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=test_split_factor, random_state=42)
    X_test = np.array(list(X_test) +  extra_fake_features)
    y_test = np.array(list(y_test) + ([1] * len(extra_fake_features)))
    discriminator = baseline_model()
    discriminator.fit(X_train, y_train, epochs=500, batch_size=16, verbose=1)
    discriminator.summary()
    plot_model(discriminator, to_file='anndiscriminator.png', show_shapes=True, show_layer_names=True)
    predictions = discriminator.predict(X_test)
    correct = 0
    wrong = 0
    predicted = []
    positives = 0
    negatives = 0

    for i in range(len(predictions)):
        predicted_class = np.argmax(predictions[i])
        predicted.append(predicted_class)
        target_class = y_test[i]
        if y_test[i] == 0:
            positives += 1
        else:
            negatives += 1
        if predicted_class == target_class:
            correct += 1
        else:
            wrong += 1

    accuracy = correct / (correct + wrong)
    print(key)
    print("Accuracy of model is ")
    print(accuracy)
    eventinferenceresults[key] = accuracy
    results = confusion_matrix(y_test, predicted)
    print("f1 score")
    f1 = f1_score(y_test, predicted, pos_label = 0, average='binary')
    print(f1)
    eventinferencef1[key] = f1
    q = positives / (positives + negatives)
    f1_coin = (2 * q)/(q + 1)
    f1_norm = (f1 - f1_coin)/(1 - f1_coin)
    eventinferencef1norm[key] = f1_norm
    print("f1 norm")
    print(f1_norm)
    print("confusion matrix")
    print(results)

    mcc = matthews_corrcoef(y_test, predicted)
    print("mcc")
    print(mcc)
    mccscores[key] = mcc

with open("patterninferenceaccuraciesexperiment0.pkl", mode='wb') as featureOutputFile:
    pickle.dump(eventinferenceresults, featureOutputFile)

with open("patterninferencef1experiment0.pkl", mode='wb') as featureOutputFile:
    pickle.dump(eventinferencef1, featureOutputFile)

with open("patterninferencef1normexperiment0.pkl", mode='wb') as featureOutputFile:
    pickle.dump(eventinferencef1norm, featureOutputFile)

with open("patterninferencemccscoresexperiment0.pkl", mode='wb') as featureOutputFile:
    pickle.dump(mccscores, featureOutputFile)