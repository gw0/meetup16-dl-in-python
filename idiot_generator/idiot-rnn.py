#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=C0103,W0621
"""
Train a text generating LSTM on Slovenian poems and prose

- first train a few epochs on Slovenian poetry and prose (to learn basics of the language) (from <http://lit.ijs.si/>)
- afterwards train at least additional epochs on target texts (to fine-tune) (from I.D.I.O.T <http://id.iot.si/>)

Based on <https://github.com/fchollet/keras/commits/master/examples/lstm_text_generation.py> and <https://karpathy.github.io/2015/05/21/rnn-effectiveness/>.
"""

from __future__ import print_function
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Embedding, LSTM
from keras.utils.visualize_util import plot
import numpy as np
import random
import os
import codecs
import re
import sys


# defaults
epochs_all = 2
epochs_target = 100
maxlen = 40
step = 3

model_yaml = "./out/model.yaml"
model_png = "./out/model.png"
weights_all_ffmt = "./out/weights_all.{}.hdf5"
weights_target_ffmt = "./out/weights_target.{}.hdf5"

# read datasets
def read_text(dir):
    text = ""
    for filename in os.listdir(dir):
        if filename.endswith(".txt"):
            f = codecs.open(os.path.join(dir, filename), 'r', encoding='utf8')
            t = f.read()
            t = re.sub('\r', '', t)
            t = re.sub('\t|    +', '    ', t)
            t = re.sub(u'…', '...', t)
            t = re.sub(u'—', '-', t)
            t = re.sub(u'»', '>', t)
            t = re.sub(u'«', '<', t)
            t = re.sub(u'’', "'", t)
            t = re.sub(u'[^A-ZČĆŠŽÄËÏÖÜa-zčćšžäëïöüß0-9 .,!?:;+-~*/$%&()<>\'\n]', '', t)
            t = re.sub('\([^ ]\) +', '\1 ', t)
            text += t
            f.close()
    print("  corpus '{}' (length {})".format(dir, len(text)))
    return text

print("read datasets...")
text = ""
text += read_text("./slovenian-poetry")
text += read_text("./slovenian-prose")
text_target = read_text("./idiot")
text += text_target

chars = set(text)
print("  total length: {}, chars: {}".format(len(text), len(chars)))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

def vectorization(text, chars, maxlen, step):
    # cut all text in semi-redundant sequences of maxlen characters
    sentences = []
    next_chars = []
    for i in range(0, len(text) - maxlen, step):
        sentences.append(text[i: i + maxlen])
        next_chars.append(text[i + maxlen])
    print("  cut sentences: {}".format(len(sentences)))

    # one-hot encoding for X and y
    #X = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
    #y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
    #for i, sentence in enumerate(sentences):
    #    for t, char in enumerate(sentence):
    #        X[i, t, char_indices[char]] = 1
    #    y[i, char_indices[next_chars[i]]] = 1

    # character embeddings for X, one-hot encoding for y
    X = np.zeros((len(sentences), maxlen), dtype=np.int32)
    y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
    for i, sentence in enumerate(sentences):
        for t, char in enumerate(sentence):
            X[i, t] = char_indices[char]
        y[i, char_indices[next_chars[i]]] = 1
    print("  shapes: {} {}".format(X.shape, y.shape))

    return X, y

print("vectorization...")
X, y = vectorization(text, chars, maxlen=maxlen, step=step)
X_target, y_target = vectorization(text_target, chars, maxlen=maxlen, step=step)

# build model
# (2 stacked LSTM)
print("build model...")
model = Sequential()
model.add(Embedding(input_dim=len(chars), output_dim=512, input_length=maxlen, mask_zero=True)
)
model.add(Dropout(0.2))
model.add(LSTM(512, return_sequences=True, input_shape=(maxlen, len(chars))))
model.add(Dropout(0.2))
model.add(LSTM(512, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(len(chars)))
model.add(Activation('softmax'))

with open(model_yaml, 'w') as f:
    model.to_yaml(stream=f)
model.summary()
plot(model, to_file=model_png, show_shapes=True)
model.compile(loss='categorical_crossentropy', optimizer='adam')

# train model on all datasets
def sample(a, temperature=1.0):
    # helper function to sample an index from a probability array
    a = np.log(a) / temperature
    a = np.exp(a) / np.sum(np.exp(a))
    return np.argmax(np.random.multinomial(1, a, 1))

print("train model on all datasets...")
for iteration in range(0, epochs_all):
    print()
    print('-' * 50)
    print('Iteration', iteration)
    if os.path.isfile(weights_all_ffmt.format(iteration)):
        model.load_weights(weights_all_ffmt.format(iteration))
        continue
    model.fit(X, y, batch_size=128, nb_epoch=1)
    model.save_weights(weights_all_ffmt.format(iteration), overwrite=True)

    # output some sample generated text
    start_index = random.randint(0, len(text) - maxlen - 1)
    for diversity in [0.2, 0.5, 1.0, 1.2]:
        print()
        print('----- diversity:', diversity)

        generated = ''
        sentence = text[start_index: start_index + maxlen]
        generated += sentence
        print(u'----- Generating with seed: "' + sentence + '"')
        sys.stdout.write(generated)

        for i in range(400):
            #x = np.zeros((1, maxlen, len(chars)))
            x = np.zeros((1, maxlen))
            for t, char in enumerate(sentence):
                #x[0, t, char_indices[char]] = 1.
                x[0, t] = char_indices[char]

            preds = model.predict(x, verbose=0)[0]
            next_index = sample(preds, diversity)
            next_char = indices_char[next_index]

            generated += next_char
            sentence = sentence[1:] + next_char

            sys.stdout.write(next_char)
            sys.stdout.flush()
        print()

print("train model on target datasets...")
for iteration in range(epochs_all, epochs_target):
    print()
    print('-' * 50)
    print('Iteration', iteration)
    if os.path.isfile(weights_target_ffmt.format(iteration)):
        model.load_weights(weights_target_ffmt.format(iteration))
        continue
    model.fit(X_target, y_target, batch_size=128, nb_epoch=1)
    model.save_weights(weights_target_ffmt.format(iteration), overwrite=True)

    # output some sample generated text
    start_index = random.randint(0, len(text) - maxlen - 1)
    for diversity in [0.2, 0.5, 1.0, 1.2]:
        print()
        print('----- diversity:', diversity)

        generated = ''
        sentence = text[start_index: start_index + maxlen]
        generated += sentence
        print(u'----- Generating with seed: "' + sentence + '"')
        sys.stdout.write(generated)

        for i in range(400):
            #x = np.zeros((1, maxlen, len(chars)))
            x = np.zeros((1, maxlen))
            for t, char in enumerate(sentence):
                #x[0, t, char_indices[char]] = 1.
                x[0, t] = char_indices[char]

            preds = model.predict(x, verbose=0)[0]
            next_index = sample(preds, diversity)
            next_char = indices_char[next_index]

            generated += next_char
            sentence = sentence[1:] + next_char

            sys.stdout.write(next_char)
            sys.stdout.flush()
        print()
