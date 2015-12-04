from __future__ import print_function
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.datasets.data_utils import get_file
from keras.models import model_from_json
import numpy as np
import random
import string
import sys
import os

'''
    Based on Keras example code: LSTM_keras.py
    According to the example, at least 20 epochs are required before the generated text starts sounding coherent. This script will select the largest existing save to load by default. If there are none, it will start to train a network to produce the appropriate output.
'''

def num2word(integ):
    return {1:'One', 2:'Two', 3:'Three', 4:'Four', 5:'Five', 6:'Six', 7:'Seven', 8:'Eight', 9:'Nine', 10:'Ten', 11:'Eleven', 12:'Twelve', 13:'Thirteen', 14:'Fourteen', 15:'Fifteen', 16:'Sixteen', 17:'Seventeen', 18:'Eighteen', 19:'Nineteen', 20:'Twenty'}[integ]

def chaptitle(integ):
    return "\n\n***\n\nChapter "+num2word(integ)+"\n\n***\n\n"

def capfirst(istring):
#    print("STRING ",istring, len(istring))
    if len(istring) == 0:
        return istring
    first, rest = istring[0], istring[1:]
    zeroth = ''
    if first == "\"" and len(rest) > 0:
        zeroth = first
        first = rest[0]
        rest = rest[1:]
    ostring = zeroth + first.capitalize() + rest
    return ostring

def start_weights(filen):
    # helper function to check if the appropriate file exists
    return filen[0:7] == 'weights'


def sample(a, temperature=1.0):
    # helper function to sample an index from a probability array
    a = np.log(a) / temperature
    a = np.exp(a) / np.sum(np.exp(a))
    return np.argmax(np.random.multinomial(1, a, 1))


# find existing files!
cwd = os.getcwd()
files = filter(start_weights,os.listdir(cwd))
startnumber = 0
for filen in files:
#    print(string.split(filen[7:],'.'))
    startnumber = max(startnumber,int(string.split(filen[7:],'.')[0]))
print('Starting at...'+str(startnumber))

if startnumber > 0:
    model = model_from_json(open('architecture'+str(startnumber)+'.json').read())
    model.load_weights('weights'+str(startnumber)+'.h5')
    maxlen = 20
    path = 'verne_2889_moon'
    text = open(path).read().lower()
    chars = set(text)
    char_indices = dict((c, i) for i, c in enumerate(chars))
    indices_char = dict((i, c) for i, c in enumerate(chars))

# build the model: 2 stacked LSTM
else:
    # TEXT SETUP
    path = 'verne_2889_moon'
    text = open(path).read().lower()
    print('corpus length:', len(text))

    chars = set(text)
    print('total chars:', len(chars))
    char_indices = dict((c, i) for i, c in enumerate(chars))
    indices_char = dict((i, c) for i, c in enumerate(chars))

    # cut the text in semi-redundant sequences of maxlen characters
    maxlen = 20
    step = 3
    sentences = []
    next_chars = []
    for i in range(0, len(text) - maxlen, step):
        sentences.append(text[i: i + maxlen])
        next_chars.append(text[i + maxlen])
    print('nb sequences:', len(sentences))

    print('Vectorization...')
    X = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
    y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
    for i, sentence in enumerate(sentences):
        for t, char in enumerate(sentence):
            X[i, t, char_indices[char]] = 1
        y[i, char_indices[next_chars[i]]] = 1
    # END TEXT SETUP 

    print('Build model...')
    model = Sequential() 
    model.add(LSTM(512, return_sequences=True, input_shape=(maxlen, len(chars))))
    model.add(Dropout(0.5)) # were 0.2
    model.add(LSTM(512, return_sequences=False))
    model.add(Dropout(0.5))
    model.add(Dense(len(chars)))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    # train the model, output generated text after each iteration
    #for iteration in range(1, 60):
    for iteration in range(int(startnumber)+1, 60):
        print()
        print('-' * 50)
        print('Iteration', iteration)
        model.fit(X, y, batch_size=128, nb_epoch=1)

        json_string = model.to_json()
        open('architecture'+str(iteration)+'.json','w').write(json_string)
        model.save_weights('weights'+str(iteration)+'.h5')

        start_index = random.randint(0, len(text) - maxlen - 1)

        for diversity in [0.2, 0.5, 1.0, 1.2]:
            print()
            print('----- diversity:', diversity)

            generated = ''
            sentence = text[start_index: start_index + maxlen]
            generated += sentence
            print('----- Generating with seed: "' + sentence + '"')
            sys.stdout.write(generated)

            for iteration in range(400):
                x = np.zeros((1, maxlen, len(chars)))
                for t, char in enumerate(sentence):
                    x[0, t, char_indices[char]] = 1.

                preds = model.predict(x, verbose=0)[0]
                next_index = sample(preds, diversity)
                next_char = indices_char[next_index]

                generated += next_char
                sentence = sentence[1:] + next_char

                sys.stdout.write(next_char)
                sys.stdout.flush()
            print()

# Produce text
start_index = random.randint(0, len(text) - maxlen - 1)
diversity = 0.7
generated = ''
sentence = text[start_index: start_index + maxlen]
for iteration in range(300000):
    x = np.zeros((1,maxlen,len(chars)))
    for t, char in enumerate(sentence):
        x[0, t, char_indices[char]] = 1.

    preds = model.predict(x, verbose=0)[0]
    next_index = sample(preds, diversity)
    next_char = indices_char[next_index]

    generated += next_char
    sentence = sentence[1:] + next_char
    
    if len(generated) % 500 == 0:
        print (str(len(generated)))

#    open('50000.txt', 'w').write(generated)
    open('300000.txt', 'w').write(generated)

import re

#source_text = open('50000_first.txt','r').read()
source_text = open('300000.txt','r').read()
source_text_edit = re.sub(r" ?\n(?!\n) ?"," ", source_text)
source_text_edit = re.sub(r"\n","\n\n",source_text_edit)
source_text_edit = re.sub("chapter [a-zA-Z0-9]{1,4} ?\n", "",source_text_edit)
source_text_edit = re.sub(r"\"\"", "\"", source_text_edit)
source_text_edit = string.join([capfirst(i.strip()) for i in source_text_edit.split(".")], ".")
source_text_edit = string.join([capfirst(i.strip()) for i in source_text_edit.split("?")], "?")
source_text_edit = string.join([capfirst(i.strip()) for i in source_text_edit.split("!")], "!")
def replacement(match):
    return "\n"+capfirst(match.group(1))
source_text_edit = re.sub(r"\n ?(\"?[a-z])", replacement, source_text_edit)
source_text_edit = re.sub(r"([.?!]\"?) *", r"\1 ", source_text_edit)
def replacement(match):
    return match.group(1) + capfirst(match.group(2))
source_text_edit = re.sub(r"([.?!]\" \")([a-z])", replacement, source_text_edit)
source_text_edit = re.sub(r"\bi\b","I",source_text_edit)
source_text_edit = re.sub(r" +"," ",source_text_edit)

#CAPITALIZE NAMES
from nltk.tag.stanford import NERTagger
st = NERTagger('stanford-ner-2014-06-16/classifiers/english.all.3class.distsim.crf.ser.gz', 'stanford-ner-2014-06-16/stanford-ner.jar')
nopunc = re.sub(r"[.?!;,_^]", " ", source_text_edit)
tagged = st.tag(nopunc.split())
def replacement(match):
    return match.group(1).capitalize()
for item in [pair[0] for pair in tagged if pair[1] == 'PERSON']: 
    source_text_edit = re.sub(r"("+item+r")", replacement, source_text_edit)
# catch some odd ones
# michel, ardan: solved by Stanford NER
for item in ['barbicane', 'nicholl','mactolla']:
    source_text_edit = re.sub(r"("+item+r")", replacement, source_text_edit)

# add chapters
chapter_split = source_text_edit.split("\n\n")
chapter_num = 1
finished_chaps = ""
accumulate = ""
for item in chapter_split:
    accumulate += "\n\n" + item
    if len(accumulate) > 25000:
        finished_chaps += chaptitle(chapter_num) + accumulate
        chapter_num += 1
        accumulate = ""
if len(accumulate) != 0:
    finished_chaps += chaptitle(chapter_num) + accumulate
source_text_edit = finished_chaps

# add title
source_text_edit = "The President of the Moon\n\n" + "Jules Vernn" + "\n\n\n" + source_text_edit

open('presidentofthemoon.txt', 'w').write(source_text_edit)
