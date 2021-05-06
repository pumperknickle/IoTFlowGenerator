import sys
import numpy as np
import pickle
from keras.models import Model, Sequential
from keras.layers import Input, Dense
from keras.optimizers import Adam
from keras.utils.vis_utils import plot_model
from sklearn.model_selection import train_test_split
from keras.layers import LSTM, Concatenate
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from utilities import get_min_duration, dbclustermin, ngrams, extractFeatures, extractSignatures, multiply_durations, concat_all, get_max_packet_size, get_max_duration, extract_packet_sizes, extract_packet_size_ranges, token_frequency_features_and_labels, packet_size_features, generate_traffic_rate_features, splitAllFeatures, extract_durations, normalize_packet_sizes
import warnings
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import os
import random
import csv
from statistics import multimode
import random

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

def extractSequences(fn):
    seqs = []
    with open(fn, newline='\n') as csvf:
        csv_reader = csv.reader(csvf, delimiter=' ')
        for row in csv_reader:
            if len(row) > 0:
                seqs.append(row)
    return seqs

iotg_generated_packets = extractSequences("final_generated_packet_sizes.txt")

with open("generated_durations.pkl", mode='rb') as tokenFile:
    iotg_flat_durations = pickle.load(tokenFile)

with open("syntheticPreprocessed.pkl", mode='rb') as featuresFile:
    dg_synthetic_features = pickle.load(featuresFile)

featuresFilePath = sys.argv[1]
target_device = "captures_master/Withings//*.pcap"

with open(featuresFilePath, mode='rb') as featuresFile:
    raw_features = pickle.load(featuresFile)

target_packet_sizes = extract_packet_sizes(raw_features[target_device])
target_durations = extract_durations(raw_features[target_device])
max_packet_size_for_target = get_max_packet_size(target_packet_sizes)
max_duration_for_target = get_max_duration(target_durations)

raw_features = splitAllFeatures(raw_features)

device_to_packet_sizes_reals = dict()
device_to_durations = dict()
device_to_timestamps_reals = dict()
device_to_traffic_feats_reals = dict()

all_keys = []
max_packet_size = 0
max_duration = 0
min_duration = 1000

for key, value in raw_features.items():
    all_keys.append(key)
    device_to_packet_sizes_reals[key] = extract_packet_sizes(value)
    max_packet_size = max(get_max_packet_size(device_to_packet_sizes_reals[key]), max_packet_size)
    print(key)
    print("max packet size")
    print(max_packet_size)
    current_durations = extract_durations(value)
    max_duration = max(get_max_duration(current_durations), max_duration)
    min_duration = min(get_min_duration(current_durations), min_duration)
    print("max duration")
    print(max_duration)
    print("min duration")
    print(min_duration)
    device_to_durations[key] = current_durations
    device_to_timestamps_reals[key] = durationsToTimestamp(current_durations)
    device_to_traffic_feats_reals[key] = generate_traffic_rate_features(concat_all(device_to_packet_sizes_reals[key], device_to_timestamps_reals[key]))
    # if key == target_device:
    #     max_packet_size_for_target = max(get_max_packet_size(device_to_packet_sizes_reals[key]), max_packet_size_for_target)
    #     max_duration_for_target = max(get_max_duration(current_durations), max_duration_for_target)

print("max size")
print(get_max_packet_size(iotg_generated_packets))
normalized_iotg_packets = normalize_packet_sizes(iotg_generated_packets, max_packet_size=(-1 * max_packet_size_for_target))[0]

print("iotg packet size max")
print(get_max_packet_size(normalized_iotg_packets))

currentIdx = 0
iotg_durations = []
for gen_packet_seq in normalized_iotg_packets:
    iotg_durations.append(iotg_flat_durations[currentIdx:currentIdx+len(gen_packet_seq)])
    currentIdx += len(gen_packet_seq)

normalized_iotg_durations = multiply_durations(iotg_durations, max_duration=max_duration_for_target)
iotg_timestamps = durationsToTimestamp(normalized_iotg_durations)
iotg_traffic_rate_feats = generate_traffic_rate_features(concat_all(normalized_iotg_packets, iotg_timestamps))

DG_packet_sizes = extract_packet_sizes(dg_synthetic_features[target_device])
DG_durations = multiply_durations(extract_durations(dg_synthetic_features[target_device]), max_duration=max_duration)
DG_timestamps = durationsToTimestamp(DG_durations)
dg_traffic_rate_feats = generate_traffic_rate_features(concat_all(DG_packet_sizes, DG_timestamps))

real_packet_sizes = device_to_packet_sizes_reals[target_device]
real_timestamps = device_to_timestamps_reals[target_device]
real_traffic_rate_feats = device_to_traffic_feats_reals[target_device]

print("reals")
print(real_packet_sizes[0:10])
print(real_timestamps[0:10])
print(real_traffic_rate_feats[0:10])

print("Iotg sample")
print(normalized_iotg_packets[0:10])
print(iotg_timestamps[0:10])
print(iotg_traffic_rate_feats[0:10])

print("DG sample")
print(DG_packet_sizes[0:10])
print(DG_timestamps[0:10])
print(dg_traffic_rate_feats[0:10])

def event_inference_model(n_classes=2):
    model = Sequential()
    model.add(Dense(300, activation='relu'))
    model.add(Dense(300, activation='relu'))
    model.add(Dense(n_classes, activation='softmax'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def packet_size_and_duration_model(tokenCount = (2 * max_packet_size) + 40, n_classes=2, max_length=20):
    in_seqs = Input(shape=(max_length,))
    in_durs = Input(shape=(max_length, 1))
    inp_s = Embedding(tokenCount + 1, 32, input_length=max_length)(in_seqs)
    merged = Concatenate(axis=2)([inp_s, in_durs])
    merged = LSTM(max_length)(merged)
    out1 = Dense(n_classes, activation='softmax')(merged)
    model = Model([in_seqs, in_durs], out1)
    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss='sparse_categorical_crossentropy', optimizer=opt)
    return model

def packet_size_model(tokenCount = (2 * max_packet_size) + 40, n_classes=2, max_length=20):
    in_seqs = Input(shape=(max_length,))
    inp_s = Embedding(tokenCount + 1, 32, input_length=max_length)(in_seqs)
    inp_s = LSTM(max_length)(inp_s)
    out1 = Dense(n_classes, activation='softmax')(inp_s)
    model = Model(in_seqs, out1)
    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss='sparse_categorical_crossentropy', optimizer=opt)
    return model

# Experiment 2 Traffic Rate

selected_features_idx = set()
for i in range(len(real_traffic_rate_feats)):
    selected = random.randint(0, len(iotg_traffic_rate_feats) - 1)
    while selected in selected_features_idx:
        selected = random.randint(0, len(iotg_traffic_rate_feats) - 1)
    selected_features_idx.add(selected)

current_metric = 100
maxMetric = 0.05
counter = 0
replace_v = 1
t_size = len(iotg_traffic_rate_feats)

def randomize(selected, replace, size):
    copySelected = selected.copy()
    sampled = random.sample(copySelected, replace)
    for samp in sampled:
        copySelected.remove(samp)
    for i in range(replace):
        select = random.randint(0, size - 1)
        while select in copySelected:
            select = random.randint(0, size - 1)
        copySelected.add(select)
    return copySelected

selected_gen_features = []
while current_metric > maxMetric:
    counter += 1
    new_selected = randomize(selected_features_idx, replace_v, t_size)
    selected_gen_features = []
    for s in new_selected:
        selected_gen_features.append(iotg_traffic_rate_feats[s])
    features = real_traffic_rate_feats + selected_gen_features
    labels = [0] * len(real_traffic_rate_feats) + [1] * len(selected_gen_features)
    features = np.array(features)
    labels = np.array(labels)
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.25, random_state=42,
                                                        stratify=labels)
    X_train_final = []
    y_train_final = []
    for i in range(len(X_train)):
        X_train_final += list(X_train[i])
        y_train_final += [y_train[i]] * len(X_train[i])

    neigh = KNeighborsClassifier(n_neighbors=5)
    neigh.fit(np.array(X_train_final), np.array(y_train_final))

    correct = 0.0

    for i in range(len(X_test)):
        test_feat = X_test[i]
        test_label = y_test[i]
        predicted = neigh.predict(test_feat)
        mm = multimode(predicted)
        if test_label in mm:
            correct += (1.0) / float(len(mm))

    correct = correct/float(len(X_test))
    result_metric = abs(0.5 - correct)
    if result_metric < current_metric:
        current_metric = result_metric
        print(correct)
        selected_features_idx = new_selected

iotg_traffic_rate_result_experiment2 = correct

features = real_traffic_rate_feats + dg_traffic_rate_feats
labels = [0] * len(real_traffic_rate_feats) + [1] * len(dg_traffic_rate_feats)
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.25, random_state=42, stratify=labels)
X_train_final = []
y_train_final = []
for i in range(len(X_train)):
    X_train_final += X_train[i]
    y_train_final += [y_train[i]] * len(X_train[i])

neigh = KNeighborsClassifier(n_neighbors=5)
neigh.fit(np.array(X_train_final), np.array(y_train_final))

correct = 0.0

for i in range(len(X_test)):
    test_feat = X_test[i]
    test_label = y_test[i]
    predicted = neigh.predict(test_feat)
    mm = multimode(predicted)
    if test_label in mm:
        correct += (1.0) / float(len(mm))

correct = correct/float(len(X_test))
dg_traffic_rate_result_experiment2 = correct

print("experiment 2 DG")
print(correct)

# distance metric used by dbscan
distance_threshold = 5.0
# total ngrams divided by cluster threshold is equal to the min_samples needed to form a cluster in dbscan
min_cluster = 4
min_sig_size = 2
max_sig_size = 5

# Experiment 1 IotG
exp1_tr_target = []
exp1_tr_rest = []
exp1_event_target = []
exp1_event_rest = []
exp1_ps_target = []
exp1_dur_target = []
exp1_ps_rest = []
exp1_dur_rest = []

signatures = dict()

for i in range(min_sig_size, max_sig_size + 1):
    allngrams = []
    for real_packet_size in real_packet_sizes:
        ngramVector = ngrams(i, real_packet_size)
        for ngram in ngramVector:
            allngrams.append(ngram)
    cluster = dbclustermin(allngrams, distance_threshold, min_cluster)
    signatures[i] = extractSignatures(cluster, i)

for device in all_keys:
    feats_for_target = device_to_packet_sizes_reals[device]
    tr_feats = device_to_traffic_feats_reals[device]
    dur_feats = device_to_durations[device]
    ps_feats = normalize_packet_sizes(feats_for_target, max_packet_size=max_packet_size)[0]
    event_feats_for_target = [None] * len(feats_for_target)
    for i in range(min_sig_size, max_sig_size + 1):
        for n in range(len(feats_for_target)):
            feature = feats_for_target[n]
            extractedNgrams = ngrams(i, feature)
            newFeatures = extractFeatures(extractedNgrams, signatures[i])
            if event_feats_for_target[n] == None:
                event_feats_for_target[n] = newFeatures
            else:
                event_feats_for_target[n] = event_feats_for_target[n] + newFeatures
    if device == target_device:
        exp1_tr_target = tr_feats
        exp1_event_target = event_feats_for_target
        exp1_dur_target = dur_feats
        exp1_ps_target = ps_feats
    else:
        exp1_tr_rest += tr_feats
        exp1_event_rest += event_feats_for_target
        exp1_ps_rest += ps_feats
        exp1_dur_rest += dur_feats

print("experiment 1")

total_desired_rest = len(exp1_event_target) * 2
if len(exp1_tr_rest) > total_desired_rest:
    exp1_tr_rest = random.sample(exp1_tr_rest, total_desired_rest)
    selected_rest = random.sample(range(0, len(exp1_ps_rest)), total_desired_rest)
    exp1_ps_rest_temp = []
    exp1_dur_rest_temp = []
    for i in selected_rest:
        exp1_ps_rest_temp.append(exp1_ps_rest[i])
        current_dur = []
        for d in exp1_dur_rest[i]:
            current_dur.append([d])
        exp1_dur_rest_temp.append(current_dur)
    exp1_ps_rest = exp1_ps_rest_temp
    exp1_dur_rest = exp1_dur_rest_temp
    exp1_event_rest = random.sample(exp1_event_rest, total_desired_rest)

exp1_dur_target_temp = []
for dur in exp1_dur_target:
    current_dur = []
    for d in dur:
        current_dur.append([d])
    exp1_dur_target_temp.append(current_dur)

exp1_dur_target = exp1_dur_target_temp

exp1_event_features = exp1_event_target + exp1_event_rest
exp1_labels = ([0] * len(exp1_event_target)) + ([1] * len(exp1_event_rest))

event_discriminator = event_inference_model()
event_discriminator.fit(np.array(exp1_event_features), np.array(exp1_labels), epochs=500, batch_size=16, verbose=1)

exp1_packet_size_features = exp1_ps_target + exp1_ps_rest
print("packet size features training shape")
print(np.array(exp1_packet_size_features).shape)
exp1_labels = ([0] * len(exp1_ps_target)) + ([1] * len(exp1_ps_rest))
print("packet size labels training shape")
print(np.array(exp1_labels).shape)

packet_size_discriminator = packet_size_model()
packet_size_discriminator.fit(np.array(exp1_packet_size_features), np.array(exp1_labels), epochs=500, batch_size=16, verbose=1)

exp1_duration_features = exp1_dur_target + exp1_dur_rest

packet_size_and_duration_discriminator = packet_size_and_duration_model()
packet_size_and_duration_discriminator.fit([np.array(exp1_packet_size_features), np.array(exp1_duration_features).astype('float32')], np.array(exp1_labels), epochs=500, batch_size=16, verbose=1)

selected_iotg_packets_features = []
selected_iotg_durations_features = []
selected_iotg_events_features = []
selected_iotg_tr_features = []
for s in selected_features_idx:
    selected_iotg_packets_features.append(normalized_iotg_packets[s])
    selected_iotg_durations_features.append(normalized_iotg_durations[s])
    selected_iotg_tr_features.append(iotg_traffic_rate_feats[s])

selected_normalized_iotg_pf = normalize_packet_sizes(selected_iotg_packets_features, max_packet_size=max_packet_size)[0]

event_feats_for_target = [None] * len(selected_iotg_packets_features)
for i in range(min_sig_size, max_sig_size + 1):
    for n in range(len(selected_iotg_packets_features)):
        feature = selected_iotg_packets_features[n]
        extractedNgrams = ngrams(i, feature)
        newFeatures = extractFeatures(extractedNgrams, signatures[i])
        if event_feats_for_target[n] == None:
            event_feats_for_target[n] = newFeatures
        else:
            event_feats_for_target[n] = event_feats_for_target[n] + newFeatures

selected_iotg_events_features = event_feats_for_target

all_iotg_packets_features = normalized_iotg_packets
all_iotg_tr_features = iotg_traffic_rate_feats

all_event_feats_for_target = [None] * len(normalized_iotg_packets)
for i in range(min_sig_size, max_sig_size + 1):
    for n in range(len(normalized_iotg_packets)):
        feature = normalized_iotg_packets[n]
        extractedNgrams = ngrams(i, feature)
        newFeatures = extractFeatures(extractedNgrams, signatures[i])
        if all_event_feats_for_target[n] == None:
            all_event_feats_for_target[n] = newFeatures
        else:
            all_event_feats_for_target[n] = all_event_feats_for_target[n] + newFeatures

all_iotg_events_features = all_event_feats_for_target

total_correct = 0
predicted = event_discriminator.predict(np.array(all_iotg_events_features))
print("event predictions")
for i in range(len(predicted)):
    predict = np.argmax(predicted[i])
    print(predicted[i])
    if predict == 0:
        total_correct += 1

exp1_iotg_event_inference = float(total_correct)/float(len(predicted))

print("Experiment 1 with event inference discriminator iotg")
print(exp1_iotg_event_inference)

all_selected_normalized_iotg_pf = normalize_packet_sizes(all_iotg_packets_features, max_packet_size=max_packet_size)[0]
predicted = packet_size_discriminator.predict(np.array(all_selected_normalized_iotg_pf))
total_correct = 0
for s in predicted:
    print(s)
    predict = np.argmax(s)
    print(predict)
    if predict == 0:
        total_correct += 1


# for i in range(len(predicted)):
#     predict = np.argmax(predicted[i])
#     print("predicted")
#     print(predict)
#     actual = selected_labels[i]
#     if predict == actual:
#         total_correct += 1

exp1_iotg_packet_size_inference = float(total_correct)/float(len(predicted))

print("Experiment 1 with packet size discriminator iotg")
print(exp1_iotg_packet_size_inference)

norm_iot_d = []
for seq in normalized_iotg_durations:
    temp_seq = []
    for element in seq:
        temp_seq.append([element])
    norm_iot_d.append(temp_seq)

predicted = packet_size_and_duration_discriminator.predict([np.array(all_selected_normalized_iotg_pf), np.array(norm_iot_d)])
total_correct = 0
for s in predicted:
    print(s)
    predict = np.argmax(s)
    print(predict)
    if predict == 0:
        total_correct += 1

exp1_iotg_packet_size_duration_inference = float(total_correct)/float(len(predicted))


print("Experiment 1 with packet size and duration discriminator iotg")
print(exp1_iotg_packet_size_duration_inference)


tr_train_X = []
tr_train_y = []

exp1_traffic_rate_features = exp1_tr_target + exp1_tr_rest
exp1_traffic_rate_labels = ([0] * len(exp1_event_target)) + ([1] * len(exp1_tr_rest))

final_traffic_rate_features_exp1 = []
final_traffic_rate_labels_exp1 = []
for i in range(len(exp1_traffic_rate_features)):
    exp1_tf_feats = exp1_traffic_rate_features[i]
    exp1_tf_labels = exp1_traffic_rate_labels[i]
    final_traffic_rate_features_exp1 += exp1_tf_feats
    final_traffic_rate_labels_exp1 += [exp1_tf_labels] * (len(exp1_tf_feats))

neigh = KNeighborsClassifier(n_neighbors=5)
neigh.fit(np.array(final_traffic_rate_features_exp1), np.array(final_traffic_rate_labels_exp1))

correct = 0.0

for i in range(len(all_iotg_tr_features)):
    test_feat = all_iotg_tr_features[i]
    predicted = neigh.predict(test_feat)
    mm = multimode(predicted)
    if 1 not in mm:
        correct += 1.0

exp1_iotg_tr = float(correct)/(len(all_iotg_tr_features))
print("Experiment 1 with traffic rate discriminator iotg")
print(exp1_iotg_tr)

correct = 0.0

for i in range(len(selected_iotg_tr_features)):
    test_feat = selected_iotg_tr_features[i]
    predicted = neigh.predict(test_feat)
    mm = multimode(predicted)
    if 1 not in mm:
        correct += 1.0

exp1_iotg_tr_selected = float(correct)/(len(selected_iotg_tr_features))
print("Experiment 1 selected with traffic rate discriminator iotg")
print(exp1_iotg_tr_selected)

# Experiment 1 DG

event_feats_for_target = [None] * len(DG_packet_sizes)
for i in range(min_sig_size, max_sig_size + 1):
    for n in range(len(DG_packet_sizes)):
        feature = DG_packet_sizes[n]
        extractedNgrams = ngrams(i, feature)
        newFeatures = extractFeatures(extractedNgrams, signatures[i])
        if event_feats_for_target[n] == None:
            event_feats_for_target[n] = newFeatures
        else:
            event_feats_for_target[n] = event_feats_for_target[n] + newFeatures

DG_events_features = event_feats_for_target

DG_labels = [0] * len(DG_events_features)

total_correct = 0
predicted = event_discriminator.predict(np.array(DG_events_features))
for i in range(len(predicted)):
    predict = np.argmax(predicted[i])
    actual = DG_labels[i]
    if predict == actual:
        total_correct += 1

exp1_DG_event_inference = float(total_correct)/float(len(predicted))

print("Experiment 1 with event inference discriminator DG")
print(exp1_DG_event_inference)

predicted = packet_size_discriminator.predict(np.array(normalize_packet_sizes(DG_packet_sizes, max_packet_size=max_packet_size)[0]))
total_correct = 0
for i in range(len(predicted)):
    predict = np.argmax(predicted[i])
    actual = DG_labels[i]
    if predict == actual:
        total_correct += 1

exp1_DG_packet_size_inference = float(total_correct)/float(len(predicted))

print("Experiment 1 with packet size discriminator DG")
print(exp1_DG_packet_size_inference)

norm_dg_d = []
for seq in DG_durations:
    temp_seq = []
    for element in seq:
        temp_seq.append([element])
    norm_dg_d.append(temp_seq)

predicted = packet_size_and_duration_discriminator.predict([np.array(normalize_packet_sizes(DG_packet_sizes, max_packet_size=max_packet_size)[0]), np.array(norm_dg_d)])
total_correct = 0
for i in range(len(predicted)):
    predict = np.argmax(predicted[i])
    actual = DG_labels[i]
    if predict == actual:
        total_correct += 1

exp1_DG_packet_size_and_duration_inference = float(total_correct)/float(len(predicted))

print("Experiment 1 with packet size and duration discriminator DG")
print(exp1_DG_packet_size_and_duration_inference)

tr_train_X = []
tr_train_y = []


final_traffic_rate_features_exp1 = []
final_traffic_rate_labels_exp1 = []
for i in range(len(exp1_traffic_rate_features)):
    exp1_tf_feats = exp1_traffic_rate_features[i]
    exp1_tf_labels = exp1_traffic_rate_labels[i]
    final_traffic_rate_features_exp1 += exp1_tf_feats
    final_traffic_rate_labels_exp1 += [exp1_tf_labels] * (len(exp1_tf_feats))

neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(np.array(final_traffic_rate_features_exp1), np.array(final_traffic_rate_labels_exp1))

correct = 0.0

for i in range(len(dg_traffic_rate_feats)):
    test_feat = dg_traffic_rate_feats[i]
    predicted = neigh.predict(test_feat)
    mm = multimode(predicted)
    if 1 not in mm:
        correct += (1.0)

exp1_DG_tr = float(correct)/(len(dg_traffic_rate_feats))
print("Experiment 1 with traffic rate discriminator DG")
print(exp1_DG_tr)

# Experiment 2 IotG PS
exp2_ps_discriminator = packet_size_model()
fake_ps_features = selected_normalized_iotg_pf
real_ps_features = normalize_packet_sizes(device_to_packet_sizes_reals[target_device], max_packet_size=max_packet_size)[0]
real_ps_labels = [0] * len(real_ps_features)
fake_ps_labels = [1] * len(fake_ps_features)

ps_features_iotg = np.array(real_ps_features + fake_ps_features)
ps_labels_iotg = np.array(real_ps_labels + fake_ps_labels)

print("fake iotg ps features")
print(fake_ps_features[0:10])
print("real ps features")
print(real_ps_features[0:10])

X_train, X_test, y_train, y_test = train_test_split(ps_features_iotg, ps_labels_iotg, test_size=0.25, random_state=42, stratify=ps_labels_iotg)
exp2_ps_discriminator.fit(X_train, y_train, epochs=500, batch_size=16, verbose=1)
predictions = exp2_ps_discriminator.predict(X_test)
correct = 0
predicted = []

for i in range(len(predictions)):
    predicted_class = np.argmax(predictions[i])
    predicted.append(predicted_class)
    target_class = y_test[i]
    if predicted_class == target_class:
        correct += 1

exp2_ps_iotg = float(correct) / float(len(predictions))
print("Experiment 2 packet size iotg")
print(exp2_ps_iotg)

real_duration_feats_unf = device_to_durations[target_device]
norm_real_d = []
for seq in real_duration_feats_unf:
    temp_seq = []
    for element in seq:
        temp_seq.append([element])
    norm_real_d.append(temp_seq)

norm_iot_d = []
for seq in selected_iotg_durations_features:
    temp_seq = []
    for element in seq:
        temp_seq.append([element])
    norm_iot_d.append(temp_seq)

dur_feats_iotg = np.array(norm_iot_d + norm_real_d)
X_train_ps, X_test_ps, X_train_dur, X_test_dur, y_train, y_test = train_test_split(ps_features_iotg, dur_feats_iotg, ps_labels_iotg, test_size=0.25, random_state=42, stratify=ps_labels_iotg)

exp2_ps_dur_discriminator = packet_size_and_duration_model()
exp2_ps_dur_discriminator.fit([X_train_ps, X_train_dur], y_train, epochs=500, batch_size=16, verbose=1)
predictions = exp2_ps_dur_discriminator.predict([X_test_ps, X_test_dur])
correct = 0
predicted = []

for i in range(len(predictions)):
    predicted_class = np.argmax(predictions[i])
    predicted.append(predicted_class)
    target_class = y_test[i]
    if predicted_class == target_class:
        correct += 1

exp2_ps_dur_iotg = float(correct) / float(len(predictions))
print("Experiment 2 packet size and duration iotg")
print(exp2_ps_dur_iotg)

# Experiment 2 DG PS
exp2_ps_discriminator = packet_size_model()
fake_ps_features = normalize_packet_sizes(DG_packet_sizes, max_packet_size=max_packet_size)[0]
real_ps_features = normalize_packet_sizes(device_to_packet_sizes_reals[target_device], max_packet_size=max_packet_size)[0]
real_ps_labels = [0] * len(real_ps_features)
fake_ps_labels = [1] * len(fake_ps_features)

ps_features_DG = np.array(real_ps_features + fake_ps_features)
ps_labels_DG = np.array(real_ps_labels + fake_ps_labels)

X_train, X_test, y_train, y_test = train_test_split(ps_features_DG, ps_labels_DG, test_size=0.25, random_state=42, stratify=ps_labels_DG)
exp2_ps_discriminator.fit(X_train, y_train, epochs=500, batch_size=16, verbose=1)
predictions = exp2_ps_discriminator.predict(X_test)
correct = 0
predicted = []

for i in range(len(predictions)):
    predicted_class = np.argmax(predictions[i])
    predicted.append(predicted_class)
    target_class = y_test[i]
    if predicted_class == target_class:
        correct += 1

exp2_ps_dg = float(correct) / float(len(predictions))
print("Experiment 2 packet size DG")
print(exp2_ps_dg)

norm_dg_d = []
for seq in DG_durations:
    temp_seq = []
    for element in seq:
        temp_seq.append([element])
    norm_dg_d.append(temp_seq)

dur_feats_dg = np.array(norm_dg_d + norm_real_d)
X_train_ps, X_test_ps, X_train_dur, X_test_dur, y_train, y_test = train_test_split(ps_features_DG, dur_feats_dg, ps_labels_DG, test_size=0.25, random_state=42, stratify=ps_labels_DG)

exp2_ps_dur_discriminator = packet_size_and_duration_model()
exp2_ps_dur_discriminator.fit([X_train_ps, X_train_dur], y_train, epochs=500, batch_size=16, verbose=1)
predictions = exp2_ps_dur_discriminator.predict([X_test_ps, X_test_dur])
correct = 0
predicted = []

for i in range(len(predictions)):
    predicted_class = np.argmax(predictions[i])
    predicted.append(predicted_class)
    target_class = y_test[i]
    if predicted_class == target_class:
        correct += 1

exp2_ps_dur_dg = float(correct) / float(len(predictions))
print("Experiment 2 packet size and duration DG")
print(exp2_ps_dur_dg)

# Experiment 2 Event Inference
exp2_ei_discriminator = event_inference_model()
fake_ei_features =  selected_iotg_events_features

feats_for_target = device_to_packet_sizes_reals[target_device]
event_feats_for_target = [None] * len(feats_for_target)
for i in range(min_sig_size, max_sig_size + 1):
    for n in range(len(feats_for_target)):
        feature = feats_for_target[n]
        extractedNgrams = ngrams(i, feature)
        newFeatures = extractFeatures(extractedNgrams, signatures[i])
        if event_feats_for_target[n] == None:
            event_feats_for_target[n] = newFeatures
        else:
            event_feats_for_target[n] = event_feats_for_target[n] + newFeatures

real_ei_labels = [0] * len(event_feats_for_target)
fake_ei_labels = [1] * len(fake_ei_features)

print("fake ei features")
print(fake_ps_features[0:10])
print("real ps features")
print(event_feats_for_target[0:10])

ei_features_iotg = np.array(event_feats_for_target + fake_ei_features)
ei_labels_iotg = np.array(real_ei_labels + fake_ei_labels)

X_train, X_test, y_train, y_test = train_test_split(ei_features_iotg, ei_labels_iotg, test_size=0.25, random_state=42, stratify=ei_labels_iotg)
exp2_ei_discriminator.fit(X_train, y_train, epochs=500, batch_size=16, verbose=1)
predictions = exp2_ei_discriminator.predict(X_test)
correct = 0
predicted = []

for i in range(len(predictions)):
    predicted_class = np.argmax(predictions[i])
    predicted.append(predicted_class)
    target_class = y_test[i]
    if predicted_class == target_class:
        correct += 1

exp2_ei_iotg = float(correct) / float(len(predictions))
print("Experiment 2 event inference iotg")
print(exp2_ei_iotg)

exp2_ei_discriminator = event_inference_model()
fake_ei_features =  DG_events_features

real_ei_labels = [0] * len(event_feats_for_target)
fake_ei_labels = [1] * len(fake_ei_features)

ei_features_dg = np.array(event_feats_for_target + fake_ei_features)
ei_labels_dg = np.array(real_ei_labels + fake_ei_labels)

X_train, X_test, y_train, y_test = train_test_split(ei_features_dg, ei_labels_dg, test_size=0.25, random_state=42, stratify=ei_labels_dg)
exp2_ei_discriminator.fit(X_train, y_train, epochs=500, batch_size=16, verbose=1)
predictions = exp2_ei_discriminator.predict(X_test)
correct = 0
predicted = []

for i in range(len(predictions)):
    predicted_class = np.argmax(predictions[i])
    predicted.append(predicted_class)
    target_class = y_test[i]
    if predicted_class == target_class:
        correct += 1

exp2_ei_dg = float(correct) / float(len(predictions))
print("Experiment 2 event inference dg")
print(exp2_ei_dg)

results_file = "exp1_packet_size_IoTG.pkl"
if os.path.isfile(results_file):
    with open(results_file, mode='rb') as tokenFile:
        results_dict = pickle.load(tokenFile)
else:
    results_dict = dict()

results_dict[target_device] = exp1_iotg_packet_size_inference

results_file = "exp1_packet_size_duration_IoTG.pkl"
if os.path.isfile(results_file):
    with open(results_file, mode='rb') as tokenFile:
        results_dict = pickle.load(tokenFile)
else:
    results_dict = dict()

results_dict[target_device] = exp1_iotg_packet_size_duration_inference

with open(results_file, mode='wb') as featureOutputFile:
    pickle.dump(results_dict, featureOutputFile)

results_file = "exp1_packet_size_DG.pkl"
if os.path.isfile(results_file):
    with open(results_file, mode='rb') as tokenFile:
        results_dict = pickle.load(tokenFile)
else:
    results_dict = dict()

results_file = "exp1_packet_size_duration_DG.pkl"
if os.path.isfile(results_file):
    with open(results_file, mode='rb') as tokenFile:
        results_dict = pickle.load(tokenFile)
else:
    results_dict = dict()

results_dict[target_device] = exp1_DG_packet_size_and_duration_inference

with open(results_file, mode='wb') as featureOutputFile:
    pickle.dump(results_dict, featureOutputFile)

results_file = "exp1_event_inference_IoTG.pkl"
if os.path.isfile(results_file):
    with open(results_file, mode='rb') as tokenFile:
        results_dict = pickle.load(tokenFile)
else:
    results_dict = dict()

results_dict[target_device] = exp1_iotg_event_inference

with open(results_file, mode='wb') as featureOutputFile:
    pickle.dump(results_dict, featureOutputFile)

results_file = "exp1_event_inference_DG.pkl"
if os.path.isfile(results_file):
    with open(results_file, mode='rb') as tokenFile:
        results_dict = pickle.load(tokenFile)
else:
    results_dict = dict()

results_dict[target_device] = exp1_DG_event_inference

with open(results_file, mode='wb') as featureOutputFile:
    pickle.dump(results_dict, featureOutputFile)

results_file = "exp1_traffic_rate_IoTG.pkl"
if os.path.isfile(results_file):
    with open(results_file, mode='rb') as tokenFile:
        results_dict = pickle.load(tokenFile)
else:
    results_dict = dict()

results_dict[target_device] = exp1_iotg_tr

with open(results_file, mode='wb') as featureOutputFile:
    pickle.dump(results_dict, featureOutputFile)

results_file = "exp1_traffic_rate_DG.pkl"
if os.path.isfile(results_file):
    with open(results_file, mode='rb') as tokenFile:
        results_dict = pickle.load(tokenFile)
else:
    results_dict = dict()

results_dict[target_device] = exp1_DG_tr

with open(results_file, mode='wb') as featureOutputFile:
    pickle.dump(results_dict, featureOutputFile)

results_file = "exp2_packet_size_IotG.pkl"
if os.path.isfile(results_file):
    with open(results_file, mode='rb') as tokenFile:
        results_dict = pickle.load(tokenFile)
else:
    results_dict = dict()

results_dict[target_device] = exp2_ps_iotg

results_file = "exp2_packet_size_duration_IotG.pkl"
if os.path.isfile(results_file):
    with open(results_file, mode='rb') as tokenFile:
        results_dict = pickle.load(tokenFile)
else:
    results_dict = dict()

results_dict[target_device] = exp2_ps_dur_iotg

with open(results_file, mode='wb') as featureOutputFile:
    pickle.dump(results_dict, featureOutputFile)

results_file = "exp2_packet_size_DG.pkl"
if os.path.isfile(results_file):
    with open(results_file, mode='rb') as tokenFile:
        results_dict = pickle.load(tokenFile)
else:
    results_dict = dict()

results_dict[target_device] = exp2_ps_dg

results_file = "exp2_packet_size_duration_DG.pkl"
if os.path.isfile(results_file):
    with open(results_file, mode='rb') as tokenFile:
        results_dict = pickle.load(tokenFile)
else:
    results_dict = dict()

results_dict[target_device] = exp2_ps_dur_dg

with open(results_file, mode='wb') as featureOutputFile:
    pickle.dump(results_dict, featureOutputFile)

results_file = "exp2_event_inference_IotG.pkl"
if os.path.isfile(results_file):
    with open(results_file, mode='rb') as tokenFile:
        results_dict = pickle.load(tokenFile)
else:
    results_dict = dict()

results_dict[target_device] = exp2_ei_iotg

with open(results_file, mode='wb') as featureOutputFile:
    pickle.dump(results_dict, featureOutputFile)

results_file = "exp2_event_inference_DG.pkl"
if os.path.isfile(results_file):
    with open(results_file, mode='rb') as tokenFile:
        results_dict = pickle.load(tokenFile)
else:
    results_dict = dict()

results_dict[target_device] = exp2_ei_dg

with open(results_file, mode='wb') as featureOutputFile:
    pickle.dump(results_dict, featureOutputFile)

results_file = "exp2_traffic_rate_IotG.pkl"
if os.path.isfile(results_file):
    with open(results_file, mode='rb') as tokenFile:
        results_dict = pickle.load(tokenFile)
else:
    results_dict = dict()

results_dict[target_device] = iotg_traffic_rate_result_experiment2

with open(results_file, mode='wb') as featureOutputFile:
    pickle.dump(results_dict, featureOutputFile)

results_file = "exp2_traffic_rate_DG.pkl"
if os.path.isfile(results_file):
    with open(results_file, mode='rb') as tokenFile:
        results_dict = pickle.load(tokenFile)
else:
    results_dict = dict()

results_dict[target_device] = dg_traffic_rate_result_experiment2

with open(results_file, mode='wb') as featureOutputFile:
    pickle.dump(results_dict, featureOutputFile)