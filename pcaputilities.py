import pyshark
import math
import statistics
from sklearn.cluster import DBSCAN, KMeans
import random
import csv
import numpy as np

def extract_all(real_packet_sizes_file):
    """
    Extract packet sequences from file of signed ints.
    Sign indicates direction
    # Arguments:
        real_packet_sizes_file: String
            path to file
    # Returns:
        normalized_packets: 2D list of unsigned ints
        V: vocab size
    """
    real_packets = extractSequences(real_packet_sizes_file)
    normalized_packets = []
    max_packet_size = 0
    for packets in real_packets:
        print(packets)
        max_packet_size = max(max([abs(int(x)) for x in packets]), max_packet_size)
    V = max_packet_size * 2
    for packets in real_packets:
        packet_sizes = [(int(x) + max_packet_size) for x in packets]
        normalized_packets.append(packet_sizes)
    return normalized_packets, V+1


def most_common(lst):
    return max(set(lst), key=lst.count)


def save_sequence(filename, sequence):
    with open(filename, 'a', newline='\n') as csvfile:
        csv_writer = csv.writer(csvfile, delimiter=' ')
        csv_writer.writerow(sequence)


def signature_sample(signature):
    samples = []
    for constraints in signature:
        sample = random.randint(constraints[0], constraints[1])
        samples.append(sample)
    return samples


def sequence_sample(sequence):
    samples = []
    for step in sequence:
        if isinstance(step, int):
            samples.append(step)
        else:
            samples = samples + signature_sample(stringToSignature(step))
    return samples


def sequences_sample(sequences):
    samples = []
    for sequence in sequences:
        samples.append(sequence_sample(sequence))
    return samples



def convert_to_timestamps(pathToFile):
    pcaps = pyshark.FileCapture(pathToFile)
    pcaps.set_debug()
    timestamps = []
    for pcap in pcaps:
        if 'IP' in pcap and 'TCP' in pcap and 'TLS' not in pcap:
            timestamps.append(float(pcap.frame_info.time_epoch))
        else:
            if 'TLS' in pcap and 'TCP' in pcap and 'IP' in pcap:
                try:
                    tlsPCAP = getattr(pcap.tls, 'tls.record.content_type')
                    if tlsPCAP == 23:
                        timestamps.append(float(pcap.frame_info.time_epoch))
                except:
                    print("TLS did not have content type attribute!")
    pcaps.close()
    return timestamps


def convert_to_durations(pathToFile):
    pcaps = pyshark.FileCapture(pathToFile)
    pcaps.set_debug()
    tuples = []
    for pcap in pcaps:
        if 'IP' in pcap and 'TCP' in pcap and 'TLS' not in pcap:
            tuples.append(float(pcap.frame_info.time_epoch))
        else:
            if 'TLS' in pcap and 'TCP' in pcap and 'IP' in pcap:
                try:
                    tlsPCAP = getattr(pcap.tls, 'tls.record.content_type')
                    if tlsPCAP == 23:
                        tuples.append(float(pcap.frame_info.time_epoch))
                except:
                    print("TLS did not have content type attribute!")
    pcaps.close()
    final_durations = []
    for i in range(len(tuples) - 1):
        final_durations.append(tuples[i + 1] - tuples[i])
    final_durations.append(0)
    return final_durations


def get_activity_order(all_sequences, all_signatures):
    signatureDictionary = dict()
    singleDictionary = dict()
    for size, signatures in all_signatures.items():
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


def all_greedy_activity_conversion(all_sequences, all_signatures):
    sorted_sigs = get_activity_order(all_sequences, all_signatures)
    all_converted = []
    for sequence in all_sequences:
        all_converted.append(greedy_activity_conversion(sequence, sorted_sigs))
    return all_converted


def extract_dictionaries_from_activities(converted):
    sigset = set()
    for c in converted:
        sigset = sigset.union(c)
    signatureToToken = {k: v for v, k in enumerate(list(sigset))}
    tokenToSignature = {v: k for k, v in signatureToToken.items()}
    return signatureToToken, tokenToSignature


def convert_range(all_ranges, all_packets, all_durations, max_duration, chunk, total_tokens, trailing=7):
    tokens = []
    durations = []
    train_X = []
    train_y = []
    all_p = []
    for i in range(len(all_ranges)):
        ranges = all_ranges[i]
        durs = all_durations[i]
        packets = all_packets[i]
        for j in range(math.floor(len(ranges)/chunk)):
            starting_chunk = j * chunk
            token = []
            pk = []
            duration_token = []
            for k in range(chunk):
                rangeToken = ranges[starting_chunk + k]
                durToken = durs[starting_chunk + k]
                packet = packets[starting_chunk + k]
                d = float(durToken)/max_duration
                token.append(rangeToken)
                duration_token.append(d)
                pk.append(packet)
            train_X.append(token)
            train_y.append(duration_token)
            all_p.append(pk)
    return train_X, train_y, all_p

def get_x_feature(prev_tokens, prev_durations, total_tokens, trailing=5):
    prev_token_len = len(prev_tokens)
    if prev_token_len > trailing:
        start = prev_token_len - trailing
        end = prev_token_len
        target_prev_tokens = prev_tokens[start:end]
        target_prev_durations = prev_durations[start:end]
        featureV = []
        for i in range(trailing):
            token = target_prev_tokens[i]
            duration = target_prev_durations[i]
            token_feature = [0] * (total_tokens + 1)
            token_feature[token] = 1
            sample = token_feature + [duration]
            featureV.append(sample)
        return featureV
    else:
        target_prev_tokens = prev_tokens
        target_prev_durations = prev_durations
        featureV = []
        for i in range(trailing-len(target_prev_tokens)):
            token_feature = [0] * (total_tokens + 1)
            token_feature[total_tokens] = 1
            sample = token_feature + [0.0]
            featureV.append(sample)
        for i in range(len(target_prev_tokens)):
            token = target_prev_tokens[i]
            duration = target_prev_durations[i]
            token_feature = [0] * (total_tokens + 1)
            token_feature[token] = 1
            sample = token_feature + [duration]
            featureV.append(sample)
        return featureV

def chunk_and_convert_ps(sequences, sig_sequences, chunk):
    all_ps = []
    all_sig = []
    for i in range(len(sig_sequences)):
        idx = 0
        sequence = sequences[i]
        sig_sequence = sig_sequences[i]
        for j in range(math.floor(len(sig_sequence)/chunk)):
            starting_chunk = j * chunk
            ps = []
            sigs = []
            for k in range(chunk):
                sig = sig_sequence[starting_chunk + k]
                sigs.append(sig)
                if isinstance(sig, int):
                    ps += sequence[idx:idx+1]
                    idx += 1
                else:
                    sig_length = len(stringToSignature(sig))
                    ps += sequence[idx:idx+sig_length]
                    idx += sig_length
            all_sig.append(sigs)
            all_ps.append(ps)
    return all_ps, all_sig


def get_training(all_tokens, tokensToSig, maxSigSize, trailing_tokens=2):
    predict_X = []
    for j in range(len(all_tokens)):
        tokens = all_tokens[j]
        print("line")
        print(j)
        for i in range(len(tokens)):
            token = int(tokens[i])
            print(token)
            sig = tokensToSig[token]
            previous_tokens = []
            if i >= trailing_tokens - 1:
                previous_tokens = tokens[i-trailing_tokens+1:i+1]
            else:
                previous_tokens = (trailing_tokens - i - 1) * [len(tokensToSig)] + tokens[0:i+1]
            cats = []
            for token in previous_tokens:
                categorical = (len(tokensToSig) + 1) * [0]
                categorical[int(token)] = 1
                cats += categorical
            if isinstance(sig, int):
                position = maxSigSize * [0]
                position[0] = 1
                final_feature = cats + position
                predict_X.append(final_feature)
            else:
                sig_length = len(stringToSignature(sig))
                for k in range(sig_length):
                    position = maxSigSize * [0]
                    position[k] = 1
                    final_feature = cats + position
                    predict_X.append(final_feature)
    return predict_X


def chunk_and_convert_to_training(signature_sequence, raw_durations, signatureToTokens, maxSigSize, trailing_tokens=2, trailing_durations=7):
    train_X = []
    train_y = []
    for i in range(len(signature_sequence)):
        signatures = signature_sequence[i]
        durations = raw_durations[i]
        duration_idx = 0
        for j in range(len(signatures)):
            previous_tokens = []
            if j >= trailing_tokens - 1:
                prev_sigs = signatures[j - trailing_tokens + 1:j+1]
                previous_tokens = [signatureToTokens[x] for x in prev_sigs]
            else:
                sig_tokens = [signatureToTokens[x] for x in signatures[0:j+1]]
                previous_tokens = (trailing_tokens - j - 1) * [len(signatureToTokens)] + sig_tokens
            cats = []
            for token in previous_tokens:
                categorical = (len(signatureToTokens) + 1) * [0]
                categorical[token] = 1
                cats += categorical
            sig = signatures[j]
            if isinstance(sig, int):
                duration = durations[duration_idx]
                if duration_idx >= trailing_durations:
                    duration_features = durations[duration_idx - trailing_durations:duration_idx]
                else:
                    end_durations = durations[0: duration_idx]
                    start_durations = [0] * (trailing_duration - duration_idx)
                    duration_features = start_durations + end_durations
                position = maxSigSize * [0]
                position[0] = 1
                final_feature = cats + position + duration_features
                train_X.append(final_feature)
                train_y.append(duration)
                duration_idx += 1
            else:
                sig_length = len(stringToSignature(sig))
                for k in range(sig_length):
                    position = maxSigSize * [0]
                    position[k] = 1
                    duration = durations[duration_idx + k]
                    if duration_idx + k >= trailing_durations:
                        duration_features = durations[duration_idx + k - trailing_durations:duration_idx + k]
                    else:
                        end_durations = durations[0: duration_idx + k]
                        start_durations = [0]  * (trailing_durations - duration_idx - k)
                        duration_features = start_durations + end_durations
                    final_feature = cats + position + duration_features
                    train_y.append(duration)
                    train_X.append(final_feature)
                duration_idx += sig_length
    return train_X, train_y


def chunk_and_convert_ps_and_durations(sequences, durations, sig_sequences, chunk):
    all_ps = []
    all_raw_duration = []
    all_duration = []
    all_sig = []
    print(sig_sequences)
    for i in range(len(sig_sequences)):
        idx = 0
        sequence = sequences[i]
        duration_sequence = durations[i]
        sig_sequence = sig_sequences[i]
        for j in range(math.floor(len(sig_sequence)/chunk)):
            starting_chunk = j * chunk
            ps = []
            raw_duration = []
            duration = []
            sigs = []
            for k in range(chunk):
                sig = sig_sequence[starting_chunk + k]
                sigs.append(sig)
                if isinstance(sig, int):
                    ps += sequence[idx:idx+1]
                    duration.append(duration_sequence[idx])
                    raw_duration.append(duration_sequence[idx])
                    idx += 1
                else:
                    sig_length = len(stringToSignature(sig))
                    ps += sequence[idx:idx+sig_length]
                    duration.append(sum(duration_sequence[idx:idx+sig_length]))
                    raw_duration += duration_sequence[idx:idx+sig_length]
                    idx += sig_length
            all_sig.append(sigs)
            all_ps.append(ps)
            all_duration.append(duration)
            all_raw_duration.append(raw_duration)
    return all_ps, all_raw_duration, all_duration, all_sig


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


def change_values(dict_to_change, old_value, new_value):
    for key, value in dict_to_change.items():
        if value == old_value:
            dict_to_change[key] = new_value


def intersect(sig1, sig2):
    return (sig1[0][0] < sig2[0][1]) and (sig2[0][0] < sig1[0][1])


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


def convertToFeatures(pathToFile):
    pcaps = pyshark.FileCapture(pathToFile)
    pcaps.set_debug()
    tuples = []
    for pcap in pcaps:
        if 'IP' in pcap and 'TCP' in pcap and 'TLS' not in pcap:
            tuples.append([pcap.ip.src, pcap.ip.dst, pcap.length])
        else:
            if 'TLS' in pcap and 'TCP' in pcap and 'IP' in pcap:
                try:
                    tlsPCAP = getattr(pcap.tls, 'tls.record.content_type')
                    if tlsPCAP == 23:
                        tuples.append([pcap.ip.src, pcap.ip.dst, pcap.length])
                except:
                    print("TLS did not have content type attribute!")
    pcaps.close()
    sources = [row[0] for row in tuples]
    destinations = [row[1] for row in tuples]
    if not sources and not destinations:
        return []
    most_common_ip = most_common(sources + destinations)
    features = []
    for row in tuples:
        if row[0] == most_common_ip:
            length = int(row[2])
            features.append(length)
        else:
            if row[1] == most_common_ip:
                length = int(row[2]) * -1
                features.append(length)
    return features


def ngrams(n, sequence):
    output = []
    for i in range(len(sequence) - n + 1):
        output.append(sequence[i:i + n])
    return output


def isPingPong(sequence):
    for i in range(len(sequence) - 1):
        if sequence[i] > 0 and sequence[i + 1] > 0:
            return False
        if sequence[i] < 0 and sequence[i + 1] < 0:
            return False
    return True


def countngrams(sequences):
    counts = dict()
    for i in sequences:
        counts[tuple(i)] = counts.get(tuple(i), 0) + 1
    return counts


def similarity(x, y, coefficient_of_variation_threshold):
    coefficients_of_variations = []
    for i in len(x):
        mean = (x.get(i, 0) + y.get(i, 0)) / 2
        variance = ((x.get(i, 0) - mean) ** 2) + ((y.get(i, 0) - mean) ** 2)
        standard_dev = math.sqrt(variance)
        coefficients_of_variations.append(float(standard_dev) / mean)
    return statistics.mean(coefficients_of_variations) < coefficient_of_variation_threshold


def computeRatio(a, b):
    if a == 0 or b == 0:
        return 1
    else:
        return 1 - (min(a/b, b/a))

def pairwise(x):
    outer = []
    for i in x:
        inner = []
        for j in x:
            inner.append(computeRatio(i, j))
        outer.append(inner)
    return outer

def evaluate_1(clusters):
    total_error = 0
    for i in range(len(clusters)):
        cluster = clusters[i]
        mean = statistics.mean(cluster)
        for element in cluster:
            if element == 0:
                total_error += abs(element - mean)
            total_error += abs(element - mean)/element
    return total_error

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
    return rangesToTokens, tokensToRanges, tokensToMean, tokensTostd

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


def dbclustermin(x, eps, min_samples):
    db = DBSCAN(eps, min_samples).fit(x)
    clusters = dict()
    for i in range(len(db.labels_)):
        if db.labels_[i] != -1:
            clusters[db.labels_[i]] = clusters.get(db.labels_[i], []) + [x[i]]
    return list(clusters.values())

# Cluster using dbscan
def dbcluster(x, eps, samples_ratio):
    min_samples = math.floor(len(x) / float(samples_ratio))
    db = DBSCAN(eps, min_samples).fit(x)
    clusters = dict()
    for i in range(len(db.labels_)):
        if db.labels_[i] != -1:
            clusters[db.labels_[i]] = clusters.get(db.labels_[i], []) + [x[i]]
    return list(clusters.values())


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


def generate_from_sig(signature):
    generated = []
    for tuple in signature:
        generated.append(random.randint(tuple[0], tuple[1]))
    return generated


def extractFeatures(ngrams, signatures):
    features = []
    for signature in signatures:
        count = 0
        for ngram in ngrams:
            if matches(ngram, signature):
                count += 1
        frequency = 0 if len(ngrams) == 0 else (count) / float(len(ngrams))
        features.append(frequency)
    return features


def signatureCount(all_signatures):
    all_sigs = 0
    for count, signatures in all_signatures.items():
        all_sigs += len(signatures)
    return all_sigs


def signatureExtractionAll(sequences, minSigSize, maxSigSize, distance_threshold, cluster_threshold):
    all_signatures = dict()
    for i in range(minSigSize, maxSigSize + 1):
        allngrams = []
        for sequence in sequences:
            ngramVector = ngrams(i, sequence)
            for ngram in ngramVector:
                allngrams.append(ngram)
        cluster = dbclustermin(allngrams, distance_threshold, cluster_threshold)
        signatures = extractSignatures(cluster, i)
        all_signatures[i] = signatures
    return all_signatures


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


def expandExtractAll(sequences, all_signatures):
    signature_features = []
    for sequence in sequences:
        signature_features = signature_features + expandAndExtract(sequence, all_signatures)
    return signature_features


def expandAndExtract(sequence, all_signatures):
    all_features = []
    counts = dict()
    for sig_length, signatures in all_signatures.items():
        counts[sig_length] = [0] * len(signatures)
    for i in range(len(sequence)):
        for sig_length, signatures in all_signatures.items():
            if sig_length <= i + 1:
                ngram = sequence[i + 1 - sig_length:i + 1]
                for j in range(len(signatures)):
                    signature = signatures[j]
                    if matches(ngram, signature):
                        counts[sig_length][j] += 1
        feature = []
        for sig_length, c in counts.items():
            v = [(float(0) if x == 0 else float(x) / float(i - sig_length + 2)) for x in c]
            feature = feature + v
        all_features.append(feature)
    return all_features


def signatureToString(signature):
    signature_ints = []
    for tuple in signature:
        signature_ints.append(tuple[0])
        signature_ints.append(tuple[1])
    return ', '.join(str(x) for x in signature_ints)

def stringToDurationSignature(item):
    item.replace(" ", "")
    arr = item.split(',')
    float_arr = [float(numeric_string) for numeric_string in arr]
    sig = []
    for i in range(0, len(float_arr), 2):
        sig.append((float_arr[i], float_arr[i + 1]))
    return sig

def stringToSignature(item):
    item.replace(" ", "")
    arr = item.split(',')
    int_arr = [int(numeric_string) for numeric_string in arr]
    sig = []
    for i in range(0, len(int_arr), 2):
        sig.append((int_arr[i], int_arr[i + 1]))
    return sig

def extractSequences(filename):
    sequences = []
    with open(filename, newline='\n') as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=' ')
        for row in csv_reader:
            sequences.append(row)
    return sequences
