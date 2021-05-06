import pickle
import numpy as np
import math
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import LeakyReLU
from keras.layers import Reshape
from numpy.random import randn
from keras.layers import Embedding
from keras.layers import Concatenate
from keras.optimizers import Adam
from keras.models import Model
from keras.layers import Input
from keras.models import Sequential
from keras.layers import Dense, LSTM, LeakyReLU
from keras.wrappers.scikit_learn import KerasClassifier
import csv
from pcaputilities import stringToDurationSignature, convertalldurations,convertalldurationstoint
from matplotlib import pyplot as plt
from tqdm import tqdm

def extractSequences(fn):
    seqs = []
    with open(fn, newline='\n') as csvf:
        csv_reader = csv.reader(csvf, delimiter=' ')
        for row in csv_reader:
            if len(row) > 0:
                seqs.append(row)
    return seqs

with open("durationRangesToToken.pkl", mode='rb') as tokenFile:
    durationRangesToToken = pickle.load(tokenFile)

with open("train_X.pkl", mode='rb') as tokenFile:
    X_train = pickle.load(tokenFile)

with open("train_y.pkl", mode='rb') as tokenFile:
    y_train = pickle.load(tokenFile)

with open("rangesToToken.pkl", mode='rb') as tokenFile:
    rangesToToken = pickle.load(tokenFile)

tokensToRanges = {v: k for k, v in rangesToToken.items()}

with open("max_duration.pkl", mode='rb') as tokenFile:
    max_duration = pickle.load(tokenFile)

def single_feature(j, sequence, durations, total_tokens, total_packet_tokens, next_packet_sizes=2, previous_packet_sizes=2, previous_durations=3):
    if len(durations) != j:
        return None
    if j >= previous_packet_sizes:
        prevTokens = sequence[j - previous_packet_sizes:j]
    else:
        prevTokens = [total_tokens] * (previous_packet_sizes - j) + sequence[:j]
    if j >= previous_durations:
        prevDurations = durations[j - previous_durations:j]
    else:
        prevDurations = [total_packet_tokens] * (previous_durations - j) + durations[:j]
    if j < len(sequence) - next_packet_sizes:
        next_tokens = sequence[j + 1:j + 1 + next_packet_sizes]
    else:
        next_tokens = sequence[j + 1:] + [total_tokens] * (j + next_packet_sizes - len(sequence) + 1)
    token = sequence[j]
    tokenVector = [int(token)] + prevTokens + next_tokens
    durationsVector = prevDurations
    return tokenVector, durationsVector

def surrounding_features(sequences, durations, total_tokens, total_packet_tokens, next_packet_sizes=2, previous_packet_sizes=2, previous_durations=3):
    print(np.array(durations).shape)
    transformed = []
    all_durations = []
    for i in range(len(sequences)):
        sequence = sequences[i]
        duration = durations[i]
        for j in range(len(sequence)):
            if j >= previous_packet_sizes:
                prevTokens = sequence[j-previous_packet_sizes:j]
            else:
                prevTokens = [total_tokens] * (previous_packet_sizes - j) + sequence[:j]
            if j >= previous_durations:
                prevDurations = duration[j-previous_durations:j]
            else:
                prevDurations = [total_packet_tokens] * (previous_durations - j) + duration[:j]
            if j < len(sequence) - next_packet_sizes:
                next_tokens = sequence[j+1:j+1+next_packet_sizes]
            else:
                next_tokens = sequence[j+1:] + [total_tokens] * (j + next_packet_sizes - len(sequence) + 1)
            token = sequence[j]
            tokenVector = [int(token)] + prevTokens + next_tokens
            durationsVector = prevDurations
            transformed.append(tokenVector)
            all_durations.append(durationsVector)
    return np.array(transformed), np.array(all_durations)

total_tokens = len(tokensToRanges)
print("total_tokens")
print(total_tokens)
print(tokensToRanges)
predict_sequences = extractSequences("tokens.txt")
predict_X = []
for i in range(len(predict_sequences)):
    sequence = predict_sequences[i]
    new_sequence = []
    for token in sequence:
        if int(token) < total_tokens:
            new_sequence.append(int(token))
        else:
            new_sequence = []
            break
    if len(new_sequence) > 0:
        predict_X.append(new_sequence)

fig = plt.figure(figsize = (10, 7))

y_train = list(y_train)
print(y_train)
new_y, tokens_to_durations = convertalldurations(y_train, durationRangesToToken)
int_new_y = convertalldurationstoint(y_train, durationRangesToToken)[0]
print(int_new_y)

y_train = np.array(new_y)
print(len(durationRangesToToken))

X_packet_train, X_duration_train = surrounding_features(list(X_train), int_new_y, total_tokens, len(durationRangesToToken))
print(X_packet_train.shape)
print(X_duration_train.shape)

# define the standalone generator model
def define_generator(latent_dim, n_classes=total_tokens, n_outputs=len(durationRangesToToken), next_packet_sizes=2, previous_packet_sizes=2, previous_durations=3):
    in_label = Input(shape=(next_packet_sizes + previous_packet_sizes + 1,))
    li = Embedding(n_classes+1, 50, input_length=next_packet_sizes + previous_packet_sizes + 1)(in_label)
    li = Flatten()(li)
    li = Dense((next_packet_sizes + previous_packet_sizes + 1) * 50)(li)
    dur_label = Input(shape=(previous_durations))
    dur = Embedding(n_outputs+1, 50, input_length=previous_durations)(dur_label)
    dur = Flatten()(dur)
    dur = Dense(previous_durations * 50)(dur)
    in_lat = Input(shape=(latent_dim,))
    gen = Dense(latent_dim)(in_lat)
    merge = Concatenate()([gen, li, dur])
    gen = Dense(1250, activation=LeakyReLU(alpha=0.1), kernel_initializer='he_uniform')(merge)
    gen = Dense(500, activation=LeakyReLU(alpha=0.1), kernel_initializer='he_uniform')(gen)
    gen = Dense(250, activation=LeakyReLU(alpha=0.1))(gen)
    out = Dense(n_outputs, activation='softmax')(gen)
    model = Model([in_lat, in_label, dur_label], out)
    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss='categorical_crossentropy', optimizer=opt)
    return model

# generate points in latent space as input for the generator
def generate_latent_points(latent_dim, labels):
    n_samples = len(labels)
    # generate points in the latent space
    x_input = randn(latent_dim * n_samples)
    # reshape into a batch of inputs for the network
    z_input = x_input.reshape(n_samples, latent_dim)
    return [z_input, np.array(labels)]

with open("tokensToMean.pkl", mode='rb') as tokenFile:
    tokensToMean = pickle.load(tokenFile)

with open("tokensToSTD.pkl", mode='rb') as tokenFile:
    tokensToSTD = pickle.load(tokenFile)


tokensToDurationRanges = {v: k for k, v in durationRangesToToken.items()}
tokenOutputs = len(tokensToDurationRanges)

latent_dim = 1
latent_train = generate_latent_points(latent_dim, X_packet_train)
print(latent_train[0].shape)
latent_predict = generate_latent_points(latent_dim, predict_X)
model = define_generator(latent_dim)
from keras.utils.vis_utils import plot_model
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
print(latent_train[0].shape)
print(latent_train[1].shape)
print(y_train.shape)
model.fit([latent_train[0], latent_train[1], np.array(X_duration_train)], y_train, batch_size = 256, epochs = 1000)
all_generated = []
print(predict_X)
for predict in tqdm(predict_X):
    generated_durations = []
    for i in range(len(predict)):
        tokenFeat, durationFeat = single_feature(i, predict, generated_durations, total_tokens, len(durationRangesToToken))
        x_input = randn(latent_dim)
        # reshape into a batch of inputs for the network
        z_input = x_input.reshape(1, latent_dim)
        pred = model.predict([z_input, np.array([tokenFeat]), np.array([durationFeat])])[0]
        durToken = np.random.choice(tokenOutputs, p=pred)
        generated_durations.append(durToken)
        prediction = np.random.choice(np.array(tokens_to_durations[durToken]))
        all_generated.append(prediction)

print(all_generated)

predicted = all_generated

with open("generated_durations.pkl", mode='wb') as sigFile:
    pickle.dump(predicted, sigFile)
