import numpy as np
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Embedding

training_samples_ = [
"nice shish kebab,", 
"amazing restaurant,", 
"would reccommend this place", 
"felt in a love!", 
"will attend this again", 
"food was crap",
"please do not visit",
"fuu",
"crap food",
"this needs new boss",
"I have got bad",
"Shawerma was perfect",
]
# Sentiment Analysis is a predictive modelling task where the model is trained 
# to predict the polarity of textual data or sentiments like Positive, Neural, and negative. 
# Also known as labels

sentiment = np.array([1,1,1,1,1,0,0,0,0,0,0,1])
# one hot encoding does encode the data into numerical values amazing became == 2, restaurant == 18
# 40 is the size of the vocabulary
test_one_hot = one_hot("horrible food", 50)
print(f"This is the test result: {test_one_hot}")
#assigning a vocabulary size
vocab_size = 50
# encoding the entire array, creating an encoded vector for each review
encoded_training_samples_ = [one_hot(d, vocab_size) for d in training_samples_]
# printing out the values
encoded_training_samples_
# Maximum length of the vector
max_length = 4
# adding a padding to shorter vectors, aka adding an extra 0
padded_training_samples_ = pad_sequences(encoded_training_samples_, maxlen=max_length, padding='post')
print(padded_training_samples_)

# creating a model
rnn_model = Sequential()

# Adding the first layer to the model
# We need to add arguments - vocab_size, vector size, the length and the label
rnn_model.add(Embedding(vocab_size, max_length, input_length=max_length, name="embedding_process"))
# Adding the second layer with flattening
rnn_model.add(Flatten())
# Applying a sigmoid activation function
rnn_model.add(Dense(1, activation="sigmoid"))

# Training set
X = padded_training_samples_
# Labels
y = sentiment

# Now we need to compile our model
# Adam is most used optimizer
# Loss Function: Binary Cross-Entropy (Output is either Yes or No == 1 or 0)
# Metrics == accuracy

rnn_model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
# printing model summary ==>

# rnn_model.summary()
# Training a model
rnn_model.fit(X, y, epochs=100, verbose=0)

# Evaluating the model
loss, accuracy = rnn_model.evaluate(X,y)

print(f"Model accuracy ==> {accuracy}")
# Getting weights
# rnn_model.get_layer("embedding_process").get_weights()[0]