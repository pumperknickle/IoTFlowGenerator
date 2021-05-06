import pickle
import numpy as np
import os
import sys
import csv
from utilities import extract_packet_sizes, splitAllFeatures, extract_durations

parentDir = "/home/joseph/generated_device/MAXGateway/results/aux_disc-False,dataset-iot,epoch-2000,epoch_checkpoint_freq-1,extra_checkpoint_freq-5,run-0,sample_len-1,self_norm-False,/sample"
featureFiles = os.listdir(parentDir)

final_files = []

for i in range(1100, 1500):
    for f in featureFiles:
        if ("epoch_id-" + str(i)) in f and "free" in f and "npz" in f:
            final_files.append(f)

print(final_files)

target_device = sys.argv[1]

with open("preprocessed.pkl", mode='rb') as featuresFile:
    raw_features = pickle.load(featuresFile)

raw_features = splitAllFeatures(raw_features)
feature_for_target = raw_features[target_device]

packet_sizes_for_target = extract_packet_sizes(feature_for_target)
durations_for_target = extract_durations(feature_for_target)
all_packet_sizes = []

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

all_syn_packet_sequences = []
all_packet_lengths = []
all_durations = []

for f in final_files:
    print(f)
    loaded_file = np.load(parentDir + "/" + f)
    for i in range(len(loaded_file['features'])):
        featureSets = loaded_file['features'][i]
        sequence = []
        packet_lengths = []
        durs = []
        for featureSet in featureSets:
            token = np.argmax(featureSet[0:len(tokensToPacketSize)])
            packet_size = tokensToPacketSize[token]
            direction = np.argmax(featureSet[len(tokensToPacketSize):len(tokensToPacketSize)+2])
            if direction == 0:
                packet_size = -1 * packet_size
            duration = featureSet[len(tokensToPacketSize)+2]
            sequence.append([packet_size, duration])
            packet_lengths.append(packet_size)
            all_durations.append(duration)
        all_syn_packet_sequences.append(sequence)
        all_packet_lengths.append(packet_lengths)

with open("generated_durations.pkl", mode='wb') as sigFile:
    pickle.dump(all_durations, sigFile)

for i in range(len(all_packet_lengths)):
  filename = 'final_generated_packet_sizes.txt'
  with open(filename, mode='a') as csvfile:
    csv_writer = csv.writer(csvfile, delimiter=' ')
    csv_writer.writerow(all_packet_lengths[i])

print(all_syn_packet_sequences)

print(final_files)