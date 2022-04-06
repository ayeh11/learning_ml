import tensorflow as tf
from tensorflow import keras
import numpy as np
import csv
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
import random

def emotion_to_num(emotion):
    switcher = {
        'A': 0,
        'H': 1,
        'Y': 2,
    }
    return switcher.get(emotion, '')

data = 'reviews4.csv'
fieldnames = ['entry', 'emotion']
temp_sentences = []
temp_labels = []
with open(data, 'r') as r:
    reader = csv.DictReader(r)
    for line in reader:
        temp_sentences.append(line['entry'])
        e = emotion_to_num(line['emotion'])
        temp_labels.append(e)

# Shuffle two lists with same order
temp = list(zip(temp_sentences, temp_labels))
random.shuffle(temp)
sentences, labels = zip(*temp)
# res1 and res2 come out as tuples, and so must be converted to lists.
sentences, labels = list(sentences), list(labels)

training_size = 600
vocab_size = 10000
embedding_dim = 16
max_length = 250
trunc_type='post'
padding_type='post'
oov_tok = "<OOV>"
batch_size = training_size

# arranging, tokenizing and padding data
training_x = sentences[0:training_size]
testing_x = sentences[training_size:]
training_y = labels[0:training_size]
testing_y = labels[training_size:]

tokenizer = Tokenizer(num_words=10000, oov_token='<OOV>')
tokenizer.fit_on_texts(training_x)

word_index = tokenizer.word_index

word_index = {k: (v+3) for k, v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2
word_index["<UNUSED>"] = 3

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()]) # flip so int:word, not word:int

training_sequences = tokenizer.texts_to_sequences(training_x)
training_padded = pad_sequences(training_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

testing_sequences = tokenizer.texts_to_sequences(testing_x)
testing_padded = pad_sequences(testing_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

training_padded = np.array(training_padded)
training_labels = np.array(training_y)
testing_padded = np.array(testing_padded)
testing_labels = np.array(testing_y)

# model
model = tf.keras.models.Sequential([tf.keras.layers.Flatten(),
                                    tf.keras.layers.Dense(128, activation=tf.nn.relu),
                                    tf.keras.layers.Dense(64, activation=tf.nn.relu),
                                    tf.keras.layers.Dense(32, activation=tf.nn.relu),
                                    tf.keras.layers.Dense(3, activation=tf.nn.softmax)])

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# batch size is how many reviews we do per cycle
fitModel = model.fit(training_padded, training_labels, epochs=50, batch_size=batch_size,
                     validation_data=(testing_padded, testing_labels), verbose=2)

results = model.evaluate(testing_padded, testing_labels)
print(f'Test loss: {results[0]} / Test accuracy: {results[1]}')

model.save("model_emotion.h5")

# learning curve
temp_train_scores = fitModel.history['accuracy']
temp_validation_scores = fitModel.history['val_accuracy']
train_scores = []
validation_scores = []
for i in range(len(temp_train_scores)):
    t = 1 - temp_train_scores[i]
    v = 1 - temp_validation_scores[i]
    train_scores.append(t)
    validation_scores.append(v)

x_axis = range(len(train_scores))

# Draw lines
plt.plot(x_axis, train_scores, '--', color="#111111",  label="Training score")
plt.plot(x_axis, validation_scores, color="#111111", label="Cross-validation score")

# Create plot
plt.title("Learning Curve")
plt.xlabel("Training Set Size"), plt.ylabel("Accuracy Score"), plt.legend(loc="best")
plt.tight_layout()
plt.show()

# tester
def review_encode(s):
    encoded = [1]
    for word in s:
        if word in word_index: # check if already in vocab
            encoded.append(word_index[word.lower()])
        else: # if not then it's a unknown
            encoded.append(2)
    return encoded

model = keras.models.load_model("model_emotion.h5")

# with open("test_phrase.txt", encoding="utf-8") as f:
#     for line in f.readlines():
#         nline = line.replace(",", "").replace(".", "").replace("(", "").replace(")", "").replace(":", "").replace("\"", "").strip().split(" ")
#         encode = review_encode(nline)
#         encode = keras.preprocessing.sequence.pad_sequences([encode], value=word_index["<PAD>"], padding="post", maxlen=250)
#         predict = model.predict(encode)
#         print(line)
#         print(encode)
#         print(predict)
#         print(np.argmax(predict))

print(testing_y)
test_review = testing_x[1]
print("Review: ")
print(test_review)
line = review_encode(test_review)
encode = keras.preprocessing.sequence.pad_sequences([line], value=word_index["<PAD>"], padding="post", maxlen=250)
predict = model.predict(encode)
print("Prediction: " + str(predict))
print(np.argmax(predict))
print("Actual: " + str(testing_labels[1]))



