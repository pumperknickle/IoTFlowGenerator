import pyshark
import math
import statistics
from sklearn.cluster import DBSCAN, KMeans
import random
import csv
import numpy as np
import glob
from tqdm import tqdm
from itertools import groupby

def sequence_features(range_tokens):
    features = []
    for key, value in tqdm(range_tokens.items()):
        for sequence in value:
            resulting_sequence = []
            for element in sequence:
                resulting_sequence.append(element)
            features.append(resulting_sequence)
    return features

def packet_size_features(packet_sizes, max_packet_size):
    features = []
    for key, value in tqdm(packet_sizes.items()):
        for sequence in value:
            resulting_sequence = []
            for element in sequence:
                resulting_sequence.append(element + max_packet_size)
            features.append(resulting_sequence)
    return features

def durationsToTimestamp(all_durations, max_duration=1.0):
    all_timestamps = []
    for durations in all_durations:
        timestamps = []
        float_durations = [float(x) for x in durations]
        for i in range(len(float_durations)):
            timestamps.append(sum(float_durations[0:i+1]) * max_duration)
        all_timestamps.append(timestamps)
    return all_timestamps

def group_then_convert(interval, packet_tuples):
    groups_iter = groupby(packet_tuples, lambda x: int(x[1]/interval))
    groups = []
    for key, group in groups_iter:
        group_array = []
        for thing in group:
            group_array.append(thing[0])
        groups.append((key, group_array))
    return groups

def group(interval, packet_tuples):
    groups_iter = groupby(packet_tuples, lambda x: int(x[1]/interval))
    groups = []
    for key, group in groups_iter:
        group_array = []
        for thing in group:
            group_array.append(thing)
        groups.append((key, group_array))
    return groups

def concat_all(all_packets, all_durations):
    result = []
    for i in range(len(all_packets)):
        packets = all_packets[i]
        durations = all_durations[i]
        if len(packets) != len(durations):
            break
        result.append(concat(packets, durations))
    return result

def concat(packets, durations):
    result = []
    for i in range(len(packets)):
        packet_size = packets[i]
        duration = durations[i]
        result.append((packet_size, duration))
    return result

def mean(lst):
    return sum(lst) / len(lst)

def splitFeatures(l, n=20):
    for i in range(0, len(l), n):
        if i+n < len(l):
            yield l[i:i+n]

def splitAllFeatures(all_features, n=20):
    features_to_return = dict()
    for key, value in all_features.items():
        all_features = []
        for sequence in value:
            all_features += splitFeatures(sequence, n)
        if len(all_features) > 20:
            features_to_return[key] = all_features
    return features_to_return

# find total traffic output through secondInterval, compute mean/std on firstInterval, create a feature vector with mean of each previous mean and std
def generate_traffic_rate_features(tuples, firstInterval = 10.0, secondIntervals = [0.1]):
    features = []
    for sequence in tuples:
        intervals = group(firstInterval, sequence)
        means = []
        stds = []
        fv = []
        for j, k in intervals:
            keyStream = k
            for secondInterval in secondIntervals:
                sub_groups = group_then_convert(secondInterval, keyStream)
                sub_group = []
                for k, v in sub_groups:
                    float_v = [abs(float(x)) for x in v]
                    sub_group.append(sum(float_v))
                sub_group = sub_group + ([0] * (int((firstInterval)/(secondInterval)) - len(sub_group)))
                means.append(mean(sub_group))
                stds.append(statistics.stdev(sub_group))
                fv.append([mean(sub_group), statistics.stdev(sub_group)])
        # featureVector = [mean(means), mean(stds)]
        # if len(stds) < 2:
        #     featureVector = [mean(means), mean(stds), 0, 0]
        # else:
        #     featureVector = [mean(means), mean(stds), statistics.stdev(means), statistics.stdev(stds)]
        features.append(fv)
    return features

def signature_frequencies_features(signature_frequencies, real=True):
    counter = 1
    features = []
    class_labels = []
    device_ids = dict()
    devices = []
    for key, value in tqdm(signature_frequencies.items()):
        for sequence in value:
            features.append(sequence)
            if real:
                class_labels.append(counter)
            else:
                class_labels.append(0)
        devices.append(key)
        device_ids[key] = counter
        counter += 1
    return features, class_labels, counter, device_ids, devices


# range_tokens is a dictionary of key device name mapped to 2d array (samples of packet size sequences)
def token_frequency_features_and_labels(range_tokens, ranges_to_tokens, real=True):
    counter = 0
    features = []
    class_labels = []
    total_tokens = len(ranges_to_tokens)
    device_ids = dict()
    print("building features")
    max_sequence_length = 0
    devices = []
    total_devices = len(ranges_to_tokens.items())
    for key, value in tqdm(range_tokens.items()):
        has_data = False
        for sequence in value:
            total_tokens_in_sequence = len(sequence)
            if total_tokens_in_sequence > max_sequence_length:
                max_sequence_length = total_tokens_in_sequence
                print(key)
                print(sequence)
            frequencies = [0.0] * total_tokens
            for token in sequence:
                frequencies[token] += 1.0/float(total_tokens_in_sequence)
            features.append(frequencies)
            if real:
                class_labels.append(counter)
                has_data = True
            else:
                class_labels.append(total_devices)
                has_data = True
        if has_data:
            devices.append(key)
        device_ids[key] = counter
        counter += 1
    return features, class_labels, counter, device_ids, max_sequence_length, devices

def stringToDurationSignature(item):
    item.replace(" ", "")
    arr = item.split(',')
    float_arr = [float(numeric_string) for numeric_string in arr]
    sig = []
    for i in range(0, len(float_arr), 2):
        sig.append((float_arr[i], float_arr[i + 1]))
    return sig

def spanSize(r):
    return r[1] - r[0]

def sortRanges(rangesToTokens):
    return sorted(rangesToTokens.items(), key=lambda x: spanSize(stringToDurationSignature(x[0])[0]))

def convertalldurations(all_durations, rangesToTokens):
    all_tokens = []
    all_tokens_to_durations = dict()
    sortedRanges = sortRanges(rangesToTokens)
    for durations in all_durations:
        tokens, tokensToDurations = convertdurations(durations, sortedRanges)
        for key, value in tokensToDurations.items():
            all_tokens_to_durations[key] = all_tokens_to_durations.get(key, []) + value
        all_tokens += tokens
    return all_tokens, all_tokens_to_durations

def convertalldurationstoint(all_durations, rangesToTokens):
    all_tokens = []
    all_tokens_to_durations = dict()
    sortedRanges = sortRanges(rangesToTokens)
    for durations in all_durations:
        tokens, tokensToDurations = convertdurationsToInt(durations, sortedRanges)
        for key, value in tokensToDurations.items():
            all_tokens_to_durations[key] = all_tokens_to_durations.get(key, []) + value
        all_tokens.append(tokens)
    return all_tokens, all_tokens_to_durations

def convertdurationsToInt(durations, sortedRanges):
    tokens = []
    tokensToDurations = dict()
    for duration in durations:
        for key, value in sortedRanges:
            signature = stringToDurationSignature(key)[0]
            if duration >= signature[0] and duration <= signature[1]:
                tokens.append(value)
                tokensToDurations[value] = tokensToDurations.get(value, []) + [duration]
                break
    return tokens, tokensToDurations

def convertdurations(durations, sortedRanges):
    tokens = []
    tokensToDurations = dict()
    for duration in durations:
        for key, value in sortedRanges:
            signature = stringToDurationSignature(key)[0]
            if duration >= signature[0] and duration <= signature[1]:
                feat = [0] * len(sortedRanges)
                feat[value] = 1
                tokens.append(feat)
                tokensToDurations[value] = tokensToDurations.get(value, []) + [duration]
                break
    return tokens, tokensToDurations

def convertallgroups(all_durations, rangesToTokens):
    merged = dict()
    sortedRanges = sortRanges(rangesToTokens)
    for durations in all_durations:
        merged = mergeall(merged, groupdurations(durations, sortedRanges))
    return merged

def convertallgroups(all_durations, rangesToTokens):
    merged = dict()
    sortedRanges = sortRanges(rangesToTokens)
    for durations in all_durations:
        merged = mergeall(merged, groupdurations(durations, sortedRanges))
    return merged

def mergeall(first, second):
    for secondKey in second.keys():
        first[secondKey] = first.get(secondKey, []) + second[secondKey]
    return first


def groupdurations(durations, sortedRanges):
    groups = dict()
    for duration in durations:
        for key, value in sortedRanges:
            signature = stringToDurationSignature(key)[0]
            if duration >= signature[0] and duration <= signature[1]:
                groups[key] = groups.get(key, []) + [duration]
    return groups

def durationcluster(x, n_clusters=20):
    x = [i for i in x if i != 0]
    newX = np.array(x)
    newX = np.log(newX)
    newX = np.expand_dims(newX, axis=1)
    clusters = dict()
    db = KMeans(n_clusters=n_clusters, random_state=1021).fit(newX)
    for i in range(len(db.labels_)):
        clusters[db.labels_[i]] = clusters.get(db.labels_[i], []) + [x[i]]
    return list(clusters.values())

def toDurationRanges(clusters):
    rangesToTokens = dict()
    tokensToRanges = dict()
    tokensToMean = dict()
    tokensTostd = dict()
    zeroRange = signatureToString([(0, 0)])
    rangesToTokens[zeroRange] = 0
    tokensToRanges[0] = zeroRange
    tokensToMean[0] = 0
    tokensTostd[0] = 0
    for i in range(len(clusters)):
        cluster = clusters[i]
        clusMin = min(cluster)
        clusMax = max(cluster)
        mean = statistics.mean(cluster)
        if len(cluster) > 1:
            std = statistics.stdev(cluster)
        else:
            std = 0
        rangeString = signatureToString([(clusMin, clusMax)])
        rangesToTokens[rangeString] = i + 1
        tokensToRanges[i + 1] = rangeString
        tokensToMean[i + 1] = mean
        tokensTostd[i + 1] = std
    return rangesToTokens, tokensToRanges

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

def extract_dictionaries_from_activities(converted):
    sigset = set()
    for c in converted:
        sigset = sigset.union(c)
    signatureToToken = {k: v for v, k in enumerate(list(sigset))}
    tokenToSignature = {v: k for k, v in signatureToToken.items()}
    return signatureToToken, tokenToSignature

def multiply_durations(durations, max_duration):
    num_seqs = []
    for sequence in durations:
        num_seq = [float(x) * max_duration for x in sequence]
        num_seqs.append(num_seq)
    return num_seqs

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

def normalize_packet_sizes(sequences, max_packet_size):
    normalized_packets = []
    for num_seq in sequences:
        normalized = [(int(x) + max_packet_size) for x in num_seq]
        normalized_packets.append(normalized)
    return normalized_packets, (max_packet_size * 2) + 1

def get_max_packet_size(sequences):
    max_packet_size = 0
    for sequence in sequences:
        num_seq = [int(x) for x in sequence]
        max_packet_size = max(max([abs(x) for x in num_seq]), max_packet_size)
    return max_packet_size

def get_max_duration(sequences):
    max_duration = 0
    for sequence in sequences:
        num_seq = [float(x) for x in sequence]
        max_duration = max(max([abs(x) for x in num_seq]), max_duration)
    return max_duration

def get_min_duration(sequences):
    min_duration = 1000.0
    for sequence in sequences:
        num_seq = [float(x) for x in sequence]
        min_duration = min(min([abs(x) for x in num_seq]), min_duration)
    return min_duration

def signatureToString(signature):
    signature_ints = []
    for tuple in signature:
        signature_ints.append(tuple[0])
        signature_ints.append(tuple[1])
    return ', '.join(str(x) for x in signature_ints)

def matches(ngram, signature):
    if len(ngram) != len(signature):
        return False
    for i in range(len(ngram)):
        ngramElement = ngram[i]
        signatureElement = signature[i]
        sigMin = signatureElement[0]
        sigMax = signatureElement[1]
        if ngramElement < sigMin or ngramElement > sigMax:
            return False
    return True

def get_activity_order(all_sequences, all_signatures):
    signatureDictionary = dict()
    singleDictionary = dict()
    print("greedily finding signature ordering")
    for size, signatures in tqdm(all_signatures.items()):
        for i in range(len(signatures)):
            signature = signatures[i]
            count = 0
            for sequence in all_sequences:
                ngramSeq = ngrams(size, sequence)
                idx = 0
                while idx <= len(ngramSeq) - size:
                    ngram = ngramSeq[idx]
                    if matches(ngram, signature):
                        count += size
                        idx += size
                    else:
                        idx += 1
            stringSig = signatureToString(signature)
            if len(signature) == 1:
                singleDictionary[stringSig] = count
            else:
                signatureDictionary[stringSig] = count
    return sorted(signatureDictionary.items(), key=lambda x: x[1], reverse=True)[0:100] + sorted(singleDictionary.items(), key=lambda x: x[1], reverse=True)[0:100]

def stringToSignature(item):
    item.replace(" ", "")
    arr = item.split(',')
    int_arr = [int(numeric_string) for numeric_string in arr]
    sig = []
    for i in range(0, len(int_arr), 2):
        sig.append((int_arr[i], int_arr[i + 1]))
    return sig

def all_greedy_activity_conversion(all_sequences, all_signatures):
    sorted_sigs = get_activity_order(all_sequences, all_signatures)
    return all_greedy_activity_conversion_sorted(all_sequences, sorted_sigs)

def all_greedy_activity_conversion_sorted(all_sequences, sorted_sigs):
    all_converted = []
    for sequence in all_sequences:
        all_converted.append(greedy_activity_conversion(sequence, sorted_sigs))
    return all_converted

def greedy_activity_conversion(sequence, sorted_signatures):
    if len(sequence) == 0:
        return []
    if len(sorted_signatures) == 0:
        return sequence
    signature_tuple = sorted_signatures[0]
    signatureString = signature_tuple[0]
    signature = stringToSignature(signatureString)
    idx = 0
    while idx <= (len(sequence) - len(signature)):
        if matches(sequence[idx:idx + len(signature)], signature):
            return greedy_activity_conversion(sequence[0:idx], sorted_signatures[1:len(sorted_signatures)]) + [
                signatureString] + greedy_activity_conversion(sequence[idx + len(signature):len(sequence)],
                                                              sorted_signatures)
        else:
            idx += 1
    return greedy_activity_conversion(sequence, sorted_signatures[1:len(sorted_signatures)])

def convert_sig_sequences_to_ranges(sig_sequences, mapping):
    range_sequences = []
    for i in range(len(sig_sequences)):
        range_sequence = []
        sig_sequence = sig_sequences[i]
        for j in range(len(sig_sequence)):
            sig = sig_sequence[j]
            if isinstance(sig, int):
                range_sequence.append(sig)
            else:
                range_sequence += convert_signatureString(sig, mapping)
        range_sequences.append(range_sequence)
    return range_sequences

def convert_signatureString(signatureString, mapping):
    signature_array = []
    sig = stringToSignature(signatureString)
    for ran in sig:
        segString = signatureToString([ran])
        signature_array.append(mapping.get(segString, segString))
    return signature_array

def change_values(dict_to_change, old_value, new_value):
    for key, value in dict_to_change.items():
        if value == old_value:
            dict_to_change[key] = new_value

def intersect(sig1, sig2):
    return (sig1[0][0] < sig2[0][1]) and (sig2[0][0] < sig1[0][1])

def majority_intersect(sig1, sig2):
    intersect = min(sig1[0][1], sig2[0][1]) - max(sig1[0][0], sig2[0][1])
    outersect = sig1[0][1] - sig1[0][0] + sig2[0][1] - sig2[0][0] - (2 * intersect)
    return intersect > outersect

def combine_signatures(sig1, sig2):
    return [(max(sig1[0][0], sig2[0][1]), min(sig1[0][1], sig2[0][1]))]

# returns mapping
def combine_all_signatures(single_signatures, evaluation_sig=None, mapping=dict()):
    if len(single_signatures) == 0:
        return mapping
    if evaluation_sig is None:
        return combine_all_signatures(single_signatures[1:], single_signatures[0], mapping)
    eval_sig = stringToSignature(evaluation_sig)
    for i in range(len(single_signatures)):
        otherSig = stringToSignature(single_signatures[i])
        if intersect(otherSig, eval_sig) and majority_intersect(otherSig, eval_sig):
            combined_signature = signatureToString(combine_signatures(otherSig, eval_sig))
            change_values(mapping, single_signatures[i], combined_signature)
            change_values(mapping, evaluation_sig, combined_signature)
            mapping[evaluation_sig] = combined_signature
            mapping[single_signatures[i]] = combined_signature
            single_signatures.pop(i)
            return combine_all_signatures(single_signatures, combined_signature, mapping)
    return combine_all_signatures(single_signatures[1:], single_signatures[0], mapping)

def map_all_signatures(all_signatures):
    single_signatures = signature_segmentation(all_signatures)
    return combine_all_signatures(single_signatures)

def signature_segmentation(all_signatures):
    single_signatures = set()
    for key, signatures in all_signatures.items():
        for sig in signatures:
            for ran in sig:
                single_signatures.add(signatureToString([ran]))
    return list(single_signatures)

def ngrams(n, sequence):
    output = []
    for i in range(len(sequence) - n + 1):
        output.append(sequence[i:i + n])
    return output

def dbclustermin(x, eps, min_samples):
    db = DBSCAN(eps, min_samples).fit(x)
    clusters = dict()
    for i in range(len(db.labels_)):
        if db.labels_[i] != -1:
            clusters[db.labels_[i]] = clusters.get(db.labels_[i], []) + [x[i]]
    return list(clusters.values())

def signatureExtractionAll(sequences, minSigSize, maxSigSize, distance_threshold, cluster_threshold):
    all_signatures = dict()
    print("extracting signatures")
    for i in tqdm(range(minSigSize, maxSigSize + 1)):
        allngrams = []
        for sequence in sequences:
            ngramVector = ngrams(i, sequence)
            for ngram in ngramVector:
                allngrams.append(ngram)
        cluster = dbclustermin(allngrams, distance_threshold, cluster_threshold)
        signatures = extractSignatures(cluster, i)
        all_signatures[i] = signatures
    return all_signatures

# Extract Signatures from cluster
def extractSignatures(clusters, n):
    signatures = []
    for cluster in clusters:
        signature = []
        for i in range(n):
            column = []
            for seq in cluster:
                column.append(seq[i])
            signature.append((min(column), max(column)))
        signatures.append(signature)
    return signatures

def most_common(lst):
    return max(set(lst), key=lst.count)

def packet_feature_extraction(pathToFile):
    pcaps = pyshark.FileCapture(pathToFile)
    features = []
    for pcap in pcaps:
        featureV = []
        if 'IP' in pcap and 'TCP' in pcap:
            if 'TLS' not in pcap:
                featureV.append(float(pcap.frame_info.time_epoch))
                featureV.append(pcap.length)
                featureV.append(pcap.ip.src)
                featureV.append(pcap.ip.dst)
                features.append(featureV)
            else:
                try:
                    tlsPCAP = getattr(pcap.tls, 'tls.record.content_type')
                    if tlsPCAP == 23:
                        featureV.append(float(pcap.frame_info.time_epoch))
                        featureV.append(pcap.length)
                        featureV.append(pcap.ip.src)
                        featureV.append(pcap.ip.dst)
                        features.append(featureV)
                except:
                    pass
    pcaps.close()
    if len(features) == 0:
        return []
    sources = [row[2] for row in features]
    destinations = [row[3] for row in features]
    most_common_ip = most_common(sources + destinations)
    final_features = []
    features.sort(key=lambda x: x[0])
    for i in range(len(features)):
        row = features[i]
        featureV = []
        duration = 0
        if i < len(features) - 1:
            nextRow = features[i+1]
            duration = nextRow[0] - row[0]
        if row[2] == most_common_ip:
            featureV.append(int(row[1]))
        else:
            if row[3] == most_common_ip:
                featureV.append(int(row[1]) * -1)
            else:
                continue
        featureV.append(duration)
        final_features.append(featureV)
    print(len(final_features))
    return final_features

def directory_packet_feature_extraction(pathToDirectory):
    extended = pathToDirectory + '/*/'
    paths = glob.glob(extended)
    features = dict()
    print("processing packet captures from " + pathToDirectory)
    print(paths)
    for i in tqdm(range(len(paths))):
        path = paths[i]
        pcapPath = path + '/*.pcap'
        pcapFiles = glob.glob(pcapPath)
        for pcapFile in pcapFiles:
            feature_extraction = packet_feature_extraction(pcapFile)
            if len(feature_extraction) > 0:
                features[pcapPath] = features.get(pcapPath, []) + [feature_extraction]
    return features

def extract_signature_frequencies(packet_sizes, min_sig_size=2, max_sig_size=5, distance_threshold=5, cluster_threshold=4):
    all_originals = []
    deviceToSignatureFrequency = dict()
    for key, value in packet_sizes.items():
        all_originals += value
    max_packet_size = get_max_packet_size(all_originals)
    all_sequences = []
    print("Normalizing Packet Sizes")
    for key, value in tqdm(packet_sizes.items()):
        all_sequences += normalize_packet_sizes(value, max_packet_size)[0]
    all_signatures = signatureExtractionAll(all_sequences, min_sig_size, max_sig_size, distance_threshold,
                                            cluster_threshold)
    print("converting signatures to features")
    numberOfSigs = 0
    for key, value in tqdm(packet_sizes.items()):
        seqs = normalize_packet_sizes(value, max_packet_size)[0]
        deviceToSignatureFrequency[key] = featureExtractionAll(seqs, all_signatures)
        if len(deviceToSignatureFrequency[key]) > 0:
            numberOfSigs = len(deviceToSignatureFrequency[key][0])
    return deviceToSignatureFrequency, numberOfSigs, max_packet_size


def featureExtractionAll(sequences, all_signatures):
  signatureFeatures = [None] * len(sequences)
  for i in range(len(sequences)):
    signatureFeatures[i] = featureExtraction(sequences[i], all_signatures)
  return signatureFeatures

def featureExtraction(sequence, all_signatures):
  all_features = []
  for i, signatures in all_signatures.items():
    ngramVector = ngrams(i, sequence)
    newFeatures = extractFeatures(ngramVector, signatures)
    all_features = all_features + newFeatures
  return all_features

def extractFeatures(ngrams, signatures):
  features = []
  for signature in signatures:
    count = 0
    for ngram in ngrams:
      if matches(ngram, signature):
        count += 1
    frequency = 0 if len(ngrams) == 0 else (count)/float(len(ngrams))
    features.append(frequency)
  return features

# Input is dictionary of device name to 2 dimensional array of packet sizes
def extract_packet_size_ranges(packet_sizes, min_sig_size=2, max_sig_size=5, distance_threshold=5, cluster_threshold=4):
    all_originals = []
    for key, value in packet_sizes.items():
        all_originals += value
    max_packet_size = get_max_packet_size(all_originals)
    all_sequences = []
    print("Normalizing Packet Sizes")
    for key, value in tqdm(packet_sizes.items()):
        if not key.startswith('fake'):
            all_sequences += normalize_packet_sizes(value, max_packet_size)[0]
    all_signatures = signatureExtractionAll(all_sequences, min_sig_size, max_sig_size, distance_threshold, cluster_threshold)
    sorted_sigs = get_activity_order(all_sequences, all_signatures)
    range_mapping = map_all_signatures(all_signatures)
    all_ranges = []
    device_to_ranges = dict()
    print("Converting 2D Array of packet sizes to 2D array of ranges")
    for key, value in tqdm(packet_sizes.items()):
        seqs = normalize_packet_sizes(value, max_packet_size)[0]
        ranges = convert_sig_sequences_to_ranges(all_greedy_activity_conversion_sorted(seqs, sorted_sigs), range_mapping)
        all_ranges += ranges
        print(ranges)
        device_to_ranges[key] = ranges
    rangesToTokens, tokenToRanges = extract_dictionaries_from_activities(all_ranges)
    packet_size_ranges = dict()
    print("Converting 2D Array of ranges to 2D array of integer tokens")
    for key, value in tqdm(device_to_ranges.items()):
        rangeSequences = []
        for seq in value:
            ranges = []
            for range in seq:
                ranges.append(rangesToTokens[range])
            rangeSequences.append(ranges)
        packet_size_ranges[key] = rangeSequences
    return packet_size_ranges, rangesToTokens, tokenToRanges, max_packet_size








