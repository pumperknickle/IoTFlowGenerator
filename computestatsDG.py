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
import statistics

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
normalized_iotg_packets = normalize_packet_sizes(iotg_generated_packets, max_packet_size=0)[0]

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

abs_packet_length_real = []
abs_packet_length_iotg = []
durations_real = list(np.array(device_to_durations[target_device]).flatten())
durations_iotg = list(np.array(normalized_iotg_durations).flatten())

for packets in real_packet_sizes:
    for packet in packets:
        abs_packet_length_real.append(abs(packet))

for packets in normalized_iotg_packets:
    for packet in packets:
        abs_packet_length_iotg.append(abs(packet))

print("average packet length real")
print(statistics.mean(abs_packet_length_real))

print("average packet length iotg")
print(statistics.mean(abs_packet_length_iotg))

print("average packet length standard deviation real")
print(statistics.stdev(abs_packet_length_real))

print("average packet length standard deviation iotg")
print(statistics.stdev(abs_packet_length_iotg))

print("average duration real")
print(statistics.mean(durations_real))

print("average duration iotg")
print(statistics.mean(durations_iotg))

print("average duration standard deviation real")
print(statistics.stdev(durations_real))

print("average duration standard deviation iotg")
print(statistics.stdev(durations_iotg))

results_file = "average_packet_lengths_dg.pkl"
if os.path.isfile(results_file):
    with open(results_file, mode='rb') as tokenFile:
        results_dict = pickle.load(tokenFile)
else:
    results_dict = dict()

results_dict[target_device] = statistics.mean(abs_packet_length_iotg)

with open(results_file, mode='wb') as featureOutputFile:
    pickle.dump(results_dict, featureOutputFile)

results_file = "stdev_packet_lengths_dg.pkl"
if os.path.isfile(results_file):
    with open(results_file, mode='rb') as tokenFile:
        results_dict = pickle.load(tokenFile)
else:
    results_dict = dict()

results_dict[target_device] = statistics.stdev(abs_packet_length_iotg)

with open(results_file, mode='wb') as featureOutputFile:
    pickle.dump(results_dict, featureOutputFile)

results_file = "average_duration_dg.pkl"
if os.path.isfile(results_file):
    with open(results_file, mode='rb') as tokenFile:
        results_dict = pickle.load(tokenFile)
else:
    results_dict = dict()

results_dict[target_device] = statistics.mean(durations_iotg)

with open(results_file, mode='wb') as featureOutputFile:
    pickle.dump(results_dict, featureOutputFile)

results_file = "stdev_duration_dg.pkl"
if os.path.isfile(results_file):
    with open(results_file, mode='rb') as tokenFile:
        results_dict = pickle.load(tokenFile)
else:
    results_dict = dict()

results_dict[target_device] = statistics.stdev(durations_iotg)

with open(results_file, mode='wb') as featureOutputFile:
    pickle.dump(results_dict, featureOutputFile)

results_file = "average_duration_real.pkl"
if os.path.isfile(results_file):
    with open(results_file, mode='rb') as tokenFile:
        results_dict = pickle.load(tokenFile)
else:
    results_dict = dict()

results_dict[target_device] = statistics.mean(durations_real)

with open(results_file, mode='wb') as featureOutputFile:
    pickle.dump(results_dict, featureOutputFile)

results_file = "stdev_duration_real.pkl"
if os.path.isfile(results_file):
    with open(results_file, mode='rb') as tokenFile:
        results_dict = pickle.load(tokenFile)
else:
    results_dict = dict()

results_dict[target_device] = statistics.stdev(durations_real)

with open(results_file, mode='wb') as featureOutputFile:
    pickle.dump(results_dict, featureOutputFile)