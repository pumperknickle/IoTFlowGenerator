import pickle
import csv
import random
from pcaputilities import sequences_sample
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.embeddings import Embedding
from keras.utils import to_categorical
import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from keras.wrappers.scikit_learn import KerasClassifier


def extractSequences(fn):
    seqs = []
    with open(fn, newline='\n') as csvf:
        csv_reader = csv.reader(csvf, delimiter=' ')
        for row in csv_reader:
            if len(row) > 0:
                seqs.append(row)
    return seqs

def stringToSignature(item):
    item.replace(" ", "")
    arr = item.split(',')
    int_arr = [int(numeric_string) for numeric_string in arr]
    sig = []
    for i in range(0, len(int_arr), 2):
        sig.append((int_arr[i], int_arr[i + 1]))
    return sig

with open("rangesToToken.pkl", mode='rb') as tokenFile:
    rangesToToken = pickle.load(tokenFile)

tokensToSignatures = {v: k for k, v in rangesToToken.items()}

#
# real_seqs = extractSequences("real_samples.txt")
#
# real_sequences = []
# for real_seq in real_seqs:
#     real_sequence = []
#     for idx in real_seq:
#         real_sequence.append(tokensToSignatures[int(idx)])
#     real_sequences.append(real_sequence)

def pad_sequence(seq, pad_to, pad_with = len(tokensToSignatures)):
    return seq + ([pad_with] * (pad_to - len(seq)))

fake_seqs = extractSequences("tokens.txt")
real_packets = extractSequences("real_packet_sizes.txt")
with open("train_X.pkl", mode='rb') as tokenFile:
    x_train = pickle.load(tokenFile)

nLefts = 4
nRights = 4

y_activity_train = []
x_activity_train = []
predict_X = []

for i in range(len(real_packets)):
    real_packet_seq = real_packets[i]
    tokens = x_train[i]
    for j in range(len(real_packet_seq)):
        x_seq = []
        current_token = int(tokens[j])
        current_real_packet = real_packet_seq[j]
        currentRange = tokensToSignatures[current_token]
        if isinstance(currentRange, int):
            y_activity_train.append(0)
        else:
            result_sig = stringToSignature(currentRange)
            y_activity_train.append(int(current_real_packet) - result_sig[0][0])
        if j < nLefts:
            stringSeq = pad_sequence(tokens[:j], nLefts)
            x_seq += [int(numeric_string) for numeric_string in stringSeq]
        else:
            stringSeq = tokens[j - nLefts:j]
            x_seq += [int(numeric_string) for numeric_string in stringSeq]
        x_seq.append(current_token)
        if (len(tokens) - j) <= nRights:
            stringSeq = pad_sequence(tokens[j+1:], nRights)
            x_seq += [int(numeric_string) for numeric_string in stringSeq]

        else:
            stringSeq = tokens[j + 1: j + nRights + 1]
            x_seq += [int(numeric_string) for numeric_string in stringSeq]
        x_activity_train.append(x_seq)

new_fake_seqs = []
y_activity_max = max(y_activity_train) + 1
max_tokens = len(tokensToSignatures)

print(tokensToSignatures)

for seq in fake_seqs:
    intSeq = [int(numeric_string) for numeric_string in seq]
    if max(intSeq) < max_tokens:
        new_fake_seqs.append(seq)

fake_seqs = new_fake_seqs

all_fake_packets = []

for i in range(len(fake_seqs)):
    tokens = fake_seqs[i]
    for j in range(len(tokens)):
        x_seq = []
        current_token = int(tokens[j])
        if j < nLefts:
            stringSeq = pad_sequence(tokens[:j], nLefts)
            x_seq += [int(numeric_string) for numeric_string in stringSeq]
        else:
            stringSeq = tokens[j - nLefts:j]
            x_seq += [int(numeric_string) for numeric_string in stringSeq]
        x_seq.append(current_token)
        if (len(tokens) - j) <= nRights:
            stringSeq = pad_sequence(tokens[j+1:], nRights)
            x_seq += [int(numeric_string) for numeric_string in stringSeq]
        else:
            stringSeq = tokens[j + 1: j + nRights + 1]
            x_seq += [int(numeric_string) for numeric_string in stringSeq]
        predict_X.append(x_seq)
        currentRange = tokensToSignatures[current_token]
        if isinstance(currentRange, int):
            all_fake_packets.append(currentRange)
        else:
            result_sig = stringToSignature(currentRange)
            all_fake_packets.append(result_sig[0][0])

encoded = to_categorical(y_activity_train)

model = Sequential()
model.add(Embedding(len(tokensToSignatures) + 1, 20, input_length=nLefts+nRights+1))
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(25, activation='relu'))
model.add(Flatten())
model.add(Dense(y_activity_max, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
print(model.summary())

X = np.array(x_activity_train)
y = np.array(to_categorical(y_activity_train))

print(X.shape)
print(y.shape)

print(np.array(predict_X).shape)

# estimator = KerasClassifier(build_fn=model, epochs=100, batch_size=1, verbose=0)
# kfold = KFold(n_splits=10, shuffle=True)
# results = cross_val_score(estimator, X, y, cv=kfold)
# print(results.mean())
# print(results.std())

model.fit(X, y, epochs=100, verbose=1)

predicted = model.predict(np.array(predict_X))
print(predicted)

all_final_pkts = []

for i in range(len(predicted)):
    prediction = predicted[i]
    print(prediction)
    fake_packet = all_fake_packets[i]
    all_final_pkts.append(np.argmax(prediction) + fake_packet)


def divide_chunks(l, n):
    # looping till length l
    for i in range(0, len(l), n):
        yield l[i:i + n]

final_fakes = list(divide_chunks(all_final_pkts, 20))


# fake_sequences = []
# for fake_seq in fake_seqs:
#     fake_sequence = []
#     maxLen = len(tokensToSignatures)
#     numer = [int(num) for num in fake_seq]
#     if max(numer) >= maxLen:
#         continue
#     print(len(fake_sequences))
#     for idx in fake_seq:
#         fake_sequence.append(tokensToSignatures[int(idx)])
#     fake_sequences.append(fake_sequence)


# fake_sequences = random.sample(fake_sequences, len(real_sequences))

# final_reals = sequences_sample(real_sequences)
# final_fakes = sequences_sample(fake_sequences)

# for i in range(len(final_reals)):
#   filename = 'exanded_real_samples.txt'
#   with open(filename, mode='a') as csvfile:
#     csv_writer = csv.writer(csvfile, delimiter=' ')
#     csv_writer.writerow(final_reals[i])

for i in range(len(final_fakes)):
  filename = 'final_generated_packet_sizes.txt'
  with open(filename, mode='a') as csvfile:
    csv_writer = csv.writer(csvfile, delimiter=' ')
    csv_writer.writerow(final_fakes[i])
