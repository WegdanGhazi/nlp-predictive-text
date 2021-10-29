# -*- coding: utf-8 -*-
"""
Created on Sat Feb  6 18:15:29 2021

@author: Wegdan
"""
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.utils import np_utils
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, LSTM, Dense, Dropout
from keras.preprocessing.text import Tokenizer
from keras.callbacks import EarlyStopping
from keras.models import Sequential
import keras.utils as ku 
from keras.models import model_from_json
from pynput import keyboard

#function to encode character inputs
def tokenize(input_string):
    tokenizer = Tokenizer(num_words=None,
                          char_level=True,
                          oov_token=None)
    corpus = input_string.lower()  
    tokenizer.fit_on_texts(corpus)
    return tokenizer

def encode(tokenizer, text):
    output = tk.texts_to_sequences([text])[0]
    output = np.array(output)/float(len(tk.index_word.keys()))
    return output

def encode_y(tokenizer, text):
    output = tk.texts_to_sequences([text])[0]
    return output

def create_model(X, y):
    model = Sequential()
    model.add(LSTM(150, return_sequences = True))
    # model.add(Dropout(0.2))
    model.add(LSTM(100))
    model.add(Dense(y.shape[1], activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    earlystop = EarlyStopping(monitor='loss', min_delta=0, patience=5, verbose=0, mode='auto')
    model.fit(X, y, epochs=20, verbose=1, callbacks=[earlystop])
    print(model.summary())
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("model.h5")
    print("Saved model to disk")
    return model

def load_model():
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("model.h5")
    print("Loaded model from disk")
    return loaded_model

def predict_next_character(X_test, tokenizer, model, max_length):
    input_length = len(X_test)
    if(input_length > max_length):
        X_test = X_test[input_length - max_length + 1 :-1]
    encoded_input = encode(tokenizer, X_test)
    encoded_input = pad_sequences([encoded_input], maxlen=max_length, padding='pre', dtype='float32')
    encoded_input = np.reshape(encoded_input, (1, max_length, 1))
    prediction = model.predict(encoded_input, verbose=0)
    index = np.argmax(prediction)
    next_char = tokenizer.sequences_to_texts([[index]])
    return next_char

### incremental prediction

def predict(input_string, tk, model, max_sequence_len):
    new_word = input_string.split()[-1] if input_string[-1] != " " and input_string else ""
    next_char = ""
    while next_char != " ":
        next_char = predict_next_character(input_string, tk, model, max_sequence_len)[0]
        input_string += next_char
        new_word += next_char
    return new_word

def on_press(key, tk, model, max_sequence_len):
    global input_string
    if key == keyboard.Key.esc:
        # Stop listener
        return False
    elif key == keyboard.Key.space:
        input_string += " "
    else:
        input_string += str(key.char)
    print(input_string)
    print(predict(input_string, tk, model, max_sequence_len))

max_sequence_len = 100

testing = True

x_data = []
y_data = []

input_data = open("84-0.txt", encoding="utf8").read()
input_data = input_data.replace('\n', '')
input_data = input_data.replace('\ufeff', '')

# this is the dictionary/tokenizer which will be used 
tk = tokenize(input_data)

if not testing:
    
    total_chars = len(input_data)
    
    #create a sliding window to go through the input
    for i in range(total_chars - max_sequence_len):
        # Define input and output sequences
        # Input is the characters within the window
        input_sec = input_data[i:i + max_sequence_len]
        # Label is the next letter up
        label = input_data[i + max_sequence_len]
    
        # We now convert list of characters to integers based on
        # previously and add the values to our lists
        x_data.append(encode(tk, input_sec))
        y_data.append(encode_y(tk, label))
    
    X = np.reshape(x_data, (len(x_data), max_sequence_len, 1))
    y = np_utils.to_categorical(y_data)
    
    model = create_model(X, y)
    
else:

    model = load_model()
    # Collect events until released
    input_string = ""
    with keyboard.Listener(on_press= lambda key: on_press(key, tk, model, max_sequence_len)) as listener:
        listener.join()
