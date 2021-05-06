from pcaputilities import convert_to_timestamps, durationcluster, toDurationRanges, convertalldurations, convert_range, convert_sig_sequences_to_ranges, map_all_signatures, chunk_and_convert_to_training, convertToFeatures, sequences_sample, chunk_and_convert_ps_and_durations, extract_dictionaries_from_activities, convert_to_durations, signatureExtractionAll, all_greedy_activity_conversion, chunk_and_convert_ps
import sys
import glob
import numpy as np
import pickle
import random
import csv

def normalize_packet_sizes(sequences):
    normalized_packets = []
    num_seqs = []
    max_packet_size = 0
    for sequence in sequences:
        num_seq = [int(x) for x in sequence]
        max_packet_size = max(max([abs(x) for x in num_seq]), max_packet_size)
        num_seqs.append(num_seq)
    for num_seq in num_seqs:
        normalized = [(x + max_packet_size) for x in num_seq]
        normalized_packets.append(normalized)
    return normalized_packets, (max_packet_size * 2) + 1


def normalize_durations(sequences):
    max_d = 0.0
    num_seqs = []
    final_num_seqs = []
    for sequence in sequences:
        num_seq = [float(x) for x in sequence]
        max_d = max(max(num_seq), max_d)
        num_seqs.append(num_seq)
    for num_seq in num_seqs:
        final_num_seq = [x/max_d for x in num_seq]
        final_num_seqs.append(final_num_seq)
    return final_num_seqs, max_d


def find_max_len(sequences):
    max_len = 0
    for sequence in sequences:
        max_len = max(len(sequence), max_len)
    return max_len

def extract_packet_sizes(sequences):
    all_packet_sizes = []
    for sequence in sequences:
        packet_sizes = []
        for feature in sequence:
            packet_size = feature[0]
            packet_sizes.append(packet_size)
        all_packet_sizes.append(packet_sizes)
    return all_packet_sizes

def extract_durations(sequences, max_duration = 1.0):
    all_durations = []
    for sequence in sequences:
        durations = []
        for feature in sequence:
            duration = float(feature[1])
            durations.append(duration/max_duration)
        all_durations.append(durations)
    return all_durations


max_duration = 0
packet_sizes = []
durations = []
labels = []

device = sys.argv[1]

directory = sys.argv[1]
extended = directory + '/*/'
paths = glob.glob(extended)

with open("preprocessed.pkl", mode='rb') as sigFile:
    all_device_flows = pickle.load(sigFile)

tuples = all_device_flows[device]

packet_sizes = extract_packet_sizes(tuples)
durations = extract_durations(tuples)

#  V is vocab size
normalized_p, V = normalize_packet_sizes(packet_sizes)

all_signatures = signatureExtractionAll(normalized_p, 2, 5, 5, 4)
range_mapping = map_all_signatures(all_signatures)
results = all_greedy_activity_conversion(normalized_p, all_signatures)
ranges = convert_sig_sequences_to_ranges(results, range_mapping)

print("normalized p")
print(normalized_p)
print("ranges")
print(ranges)
signatureToTokens, tokensToSignatures = extract_dictionaries_from_activities(results)
rangesToTokens, tokensToRanges = extract_dictionaries_from_activities(ranges)

V = len(tokensToSignatures)

with open("sigToToken.pkl", mode='wb') as sigFile:
    pickle.dump(signatureToTokens, sigFile)
with open("tokenToSig.pkl", mode='wb') as tokenFile:
    pickle.dump(tokensToSignatures, tokenFile)

with open("rangesToToken.pkl", mode='wb') as rangeFile:
    pickle.dump(rangesToTokens, rangeFile)
with open("tokensToRanges.pkl", mode='wb') as rangeFile:
    pickle.dump(tokensToRanges, rangeFile)

rangeSequences = []
for sequence in ranges:
    rans = []
    for ran in sequence:
        rans.append(rangesToTokens[ran])
    rangeSequences.append(rans)

seq_length = 20

sequences = []
for sequence in results:
    sigs = []
    for token in sequence:
        sigs.append(signatureToTokens[token])
    sequences.append(sigs)

max_duration = 0

for i in durations:
    for j in i:
        max_duration = max(max_duration, j)

minDicts = dict()
maxDicts = dict()

minDicts[0] = 10000000
maxDicts[0] = 0

def divide_chunks(l, n):
    # looping till length l
    for i in range(0, len(l), n):
        yield l[i:i + n]

all_chunks = []
all_altered_chunks = []

for i in range(len(rangeSequences)):
  filename = 'real_data.txt'
  with open(filename, mode='a') as csvfile:
    csv_writer = csv.writer(csvfile, delimiter=' ')
    chunks = divide_chunks(rangeSequences[i], seq_length)
    for chunk in chunks:
      all_chunks.append(chunk)
      if len(chunk) == seq_length:
        new_list = [x for x in chunk]
        csv_writer.writerow(new_list)


def extractSequences(fn):
    seqs = []
    with open(fn, newline='\n') as csvf:
        csv_reader = csv.reader(csvf, delimiter=' ')
        for row in csv_reader:
            seqs.append(row)
    return seqs

train_X_ranges, t_y, real_packets = convert_range(rangeSequences, normalized_p, durations, max_duration, 20, len(rangesToTokens))

for i in range(len(real_packets)):
    filename = 'real_packet_sizes.txt'
    with open(filename, mode='a') as csvfile:
        csv_writer = csv.writer(csvfile, delimiter=' ')
        chunks = divide_chunks(real_packets[i], seq_length)
        for chunk in chunks:
            if len(chunk) == seq_length:
                new_list = [x for x in chunk]
                csv_writer.writerow(new_list)

print(np.array(train_X_ranges).shape)
print(np.array(t_y).shape)

clusters = durationcluster(list(np.array(t_y).flatten()))
durationRangesToTokens, tokensToDurationRanges, tokenstoMean, tokensTostd = toDurationRanges(clusters)
with open("durationRangesToToken.pkl", mode='wb') as rangeFile:
    pickle.dump(durationRangesToTokens, rangeFile)
with open("tokensToDurationRanges.pkl", mode='wb') as rangeFile:
    pickle.dump(tokensToDurationRanges, rangeFile)

print("Max Duration")
print(max_duration)

with open("train_X.pkl", mode='wb') as sigFile:
    pickle.dump(train_X_ranges, sigFile)
with open("train_y.pkl", mode='wb') as tokenFile:
    pickle.dump(t_y, tokenFile)
with open("max_duration.pkl", mode='wb') as tokenFile:
    pickle.dump(max_duration, tokenFile)
