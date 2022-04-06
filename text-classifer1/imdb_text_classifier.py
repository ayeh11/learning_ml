import tensorflow as tf
from tensorflow import keras
import numpy as np

data = keras.datasets.imdb

# modify the default parameters of np.load
np.load.__defaults__=(None, True, True, 'ASCII')

# call load_data with allow_pickle implicitly set to true
(train_data, train_labels), (test_data, test_labels) = data.load_data(num_words=88000)

# restore np.load for future normal usage
np.load.__defaults__=(None, False, True, 'ASCII')

# data is not a bunch of numbers that are references to the actual words
word_index = data.get_word_index() # gets the index of words

word_index = {k: (v+3) for k, v in word_index.items()}
word_index["<PAD>"] = 0 # padding, is so all movie reviews are same length
word_index["<START>"] = 1
word_index["<UNK>"] = 2 # unknown
word_index["<UNUSED>"] = 3

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
# wants to switch so the int is pointing to a word and not the other way around

train_data = keras.preprocessing.sequence.pad_sequences(train_data, value=word_index["<PAD>"], padding="post", maxlen=250)
test_data = keras.preprocessing.sequence.pad_sequences(test_data, value=word_index["<PAD>"], padding="post", maxlen=250)

def decode_review(text):
    return " ".join([reverse_word_index.get(i, "?") for i in text])

# model down here
model = keras.Sequential()
model.add(keras.layers.Embedding(88000, 16))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation="relu"))
model.add(keras.layers.Dense(1, activation="sigmoid"))

model.summary()

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]) # loss is binary cuz its 0 or 1

x_val = train_data[:10000]
x_train = train_data[10000:]

y_val = train_labels[:10000]
y_train = train_labels[10000:]

# batch size is how many reviews we do per cycle
fitModel = model.fit(x_train, y_train, epochs=40, batch_size=512, validation_data=(x_val, y_val), verbose=1)

results = model.evaluate(test_data, test_labels)
print(results)

model.save("model.h5")


def review_encode(s):
    encoded = [1]
    for word in s:
        if word in word_index: # check if already in vocab
            encoded.append(word_index[word.lower()])
        else: # if not then it's a unknown
            encoded.append(2)

    return encoded

model = keras.models.load_model("model.h5")

with open("test_phrase.txt", encoding="utf-8") as f:
    for line in f.readlines():
        nline = line.replace(",", "").replace(".", "").replace("(", "").replace(")", "").replace(":", "").replace("\"", "").strip().split(" ")
        encode = review_encode(nline)
        encode = keras.preprocessing.sequence.pad_sequences([encode], value=word_index["<PAD>"], padding="post", maxlen=250)
        predict = model.predict(encode)
        print(line)
        print(encode)
        print(predict[0])



'''test_review = test_data[0]
predict = model.predict([test_review])
print("Review: ")
print(decode_review(test_review))
print("Prediction: " + str(predict[0]))
print("Actual: " + str(test_labels[0]))
print(results)'''
