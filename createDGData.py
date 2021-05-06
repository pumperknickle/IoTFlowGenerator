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
from utilities import multiply_durations, concat_all, get_max_packet_size, extractFeatures, extract_packet_sizes, ngrams, dbclustermin, extractSignatures, generate_traffic_rate_features, splitAllFeatures, extract_durations, normalize_packet_sizes
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
from gan.output import Output, OutputType, Normalization

def extractSequences(fn):
    seqs = []
    with open(fn, newline='\n') as csvf:
        csv_reader = csv.reader(csvf, delimiter=' ')
        for row in csv_reader:
            if len(row) == 20:
                seqs.append(row)
    return seqs


sys.setrecursionlimit(1000000)

warnings.filterwarnings('ignore')

# fix random seed for reproducibility
np.random.seed(7)

target_device = sys.argv[1]

with open("preprocessed.pkl", mode='rb') as featuresFile:
    raw_features = pickle.load(featuresFile)

raw_features = splitAllFeatures(raw_features)
feature_for_target = raw_features[target_device]

packet_sizes_for_target = extract_packet_sizes(feature_for_target)
durations_for_target = extract_durations(feature_for_target)

signatures = dict()

# distance metric used by dbscan
distance_threshold = 5.0
# total ngrams divided by cluster threshold is equal to the min_samples needed to form a cluster in dbscan
min_cluster = len(packet_sizes_for_target)/5
min_sig_size = 2
max_sig_size = 5
all_sigs_count = 0

for i in range(min_sig_size, max_sig_size + 1):
    allngrams = []
    for real_packet_size in packet_sizes_for_target:
        ngramVector = ngrams(i, real_packet_size)
        for ngram in ngramVector:
            allngrams.append(ngram)
    cluster = dbclustermin(allngrams, distance_threshold, min_cluster)
    signatures[i] = extractSignatures(cluster, i)
    all_sigs_count += len(signatures[i])

all_packet_sizes = []

print(all_sigs_count)

for packet_sizes in packet_sizes_for_target:
    for packet in packet_sizes:
        all_packet_sizes.append(abs(int(packet)))

all_ps_listed = list(set(all_packet_sizes))
all_ps_listed.sort()

tokensToPacketSize = dict()
packetSizeToTokens = dict()

min_duration = 1000000
max_duration = 0

for i in range(len(all_ps_listed)):
    ps = all_ps_listed[i]
    tokensToPacketSize[i] = ps
    packetSizeToTokens[ps] = i

for i in range(len(durations_for_target)):
    for dur in durations_for_target[i]:
        min_duration = min(min_duration, dur)
        max_duration = max(max_duration, dur)

data_feature = []
data_attribute = []
data_gen_flag = []

for i in range(len(packet_sizes_for_target)):
    durations = durations_for_target[i]
    packets = packet_sizes_for_target[i]
    data_gen = []
    data_feat = []
    data_attr = []
    for j in range(min_sig_size, max_sig_size + 1):
        extractedNgrams = ngrams(j, packets)
        newFeatures = extractFeatures(extractedNgrams, signatures[j])
        data_attr = data_attr + newFeatures
    for j in range(len(packets)):
        packet_length = int(packets[j])
        duration = durations[j]
        direction = []
        if packet_length < 0:
            direction = [1.0, 0.0]
        else:
            direction = [0.0, 1.0]
        token = packetSizeToTokens[abs(packet_length)]
        packet_length_feat = len(tokensToPacketSize) * [0]
        packet_length_feat[token] = 1
        normalized_duration = (duration - min_duration)/(max_duration - min_duration)
        data_gen.append(1.0)
        d = []
        d = d + packet_length_feat + direction
        d.append(normalized_duration)
        data_feat.append(np.array(d, dtype="float32"))
    data_gen_flag.append(np.array(data_gen, dtype="float32"))
    data_feature.append(np.array(data_feat))
    data_attribute.append(np.array(data_attr, dtype="float32"))

data_feature_output = [
    Output(type_=OutputType.DISCRETE, dim=len(tokensToPacketSize), normalization=None, is_gen_flag=False),
    Output(type_=OutputType.DISCRETE, dim=2, normalization=None, is_gen_flag=False),
    Output(type_=OutputType.CONTINUOUS, dim=1, normalization=Normalization.ZERO_ONE, is_gen_flag=False)
]

data_attribute_output = [
    Output(type_=OutputType.CONTINUOUS, dim=all_sigs_count, normalization=Normalization.ZERO_ONE, is_gen_flag=False),
]

data_feature = np.array(data_feature)
print(data_feature.shape)
data_attribute = np.array(data_attribute)
print(data_attribute.shape)
data_gen_flag = np.array(data_gen_flag)
print(data_gen_flag.shape)

np.savez("data/iot/data_train.npz", data_feature=data_feature, data_attribute=data_attribute, data_gen_flag=data_gen_flag)
with open('data/iot/data_feature_output.pkl', mode='wb') as fp:
    pickle.dump(data_feature_output, fp, protocol=2)
with open('data/iot/data_attribute_output.pkl', mode='wb') as fp:
    pickle.dump(data_attribute_output, fp, protocol=2)