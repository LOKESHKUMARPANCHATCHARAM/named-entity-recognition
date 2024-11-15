# Named Entity Recognition

## AIM

To develop an LSTM-based model for recognizing the named entities in the text.

## Problem Statement and Dataset
The goal of this project is to create a Named Entity Recognition (NER) model that accurately identifies and classifies entities in a text dataset by preprocessing the data, building a Bidirectional LSTM architecture, training the model, and evaluating its performance on unseen data.

![image](https://github.com/user-attachments/assets/6c8582cd-1d89-43c0-ac5a-93d31f0e2ea9)



## DESIGN STEPS

STEP 1:
Load and preprocess the NER dataset, handling missing values and mapping unique words and tags.

STEP 2:
Define a Bidirectional LSTM model with embedding and output layers for sequence labeling.

STEP 3:
Split the dataset into training and testing sets, then compile and train the model on the training data.

STEP 4:
Evaluate model performance by plotting accuracy and loss metrics over the training epochs.

STEP 5:
Make predictions on test samples and compare predicted tags against true tags for analysis.

## PROGRAM
### Name:LOKESH KUMAR P
### Register Number: 212222240054

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing import sequence
from keras import layers
from keras.models import Model
# Load dataset
data = pd.read_csv("/content/ner_dataset.csv", encoding="latin1")
data.head(15)

# Forward fill missing values
data = data.fillna(method="ffill")
data.head(50)

# Print unique words and tags in the corpus
print("Unique words in corpus:", data['Word'].nunique())
print("Unique tags in corpus:", data['Tag'].nunique())
words = list(data['Word'].unique())
words.append("ENDPAD")  # Adding a padding token
num_words = len(words)
tags = list(data['Tag'].unique())
num_tags = len(tags)

print("Unique tags are:", tags)
class SentenceGetter(object):
    def __init__(self, data):
        self.n_sent = 1
        self.data = data
        self.empty = False
        agg_func = lambda s: [(w, p, t) for w, p, t in zip(s["Word"].values.tolist(),
                                                           s["POS"].values.tolist(),
                                                           s["Tag"].values.tolist())]
        self.grouped = self.data.groupby("Sentence #").apply(agg_func)
        self.sentences = [s for s in self.grouped]
    
    def get_next(self):
        try:
            s = self.grouped["Sentence: {}".format(self.n_sent)]
            self.n_sent += 1
            return s
        except:
            return None
getter = SentenceGetter(data)
sentences = getter.sentences
len(sentences)
# Create word and tag indices
word2idx = {w: i + 1 for i, w in enumerate(words)}
tag2idx = {t: i for i, t in enumerate(tags)}

# Prepare input sequences
X1 = [[word2idx[w[0]] for w in s] for s in sentences]

# Padding sequences
max_len = 100
X = sequence.pad_sequences(maxlen=max_len,
                  sequences=X1, padding="post",
                  value=num_words-1)

# Prepare output sequences
y1 = [[tag2idx[w[2]] for w in s] for s in sentences]
y = sequence.pad_sequences(maxlen=max_len,
                  sequences=y1,
                  padding="post",
                  value=tag2idx["O"])
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2, random_state=1)
input_word = layers.Input(shape=(max_len,))

# Create the embedding layer
embedding_layer = layers.Embedding(input_dim=num_words, output_dim=50,
                                   input_length=max_len)(input_word)

# Add dropout
dropout = layers.SpatialDropout1D(0.1)(embedding_layer)

# Add a Bidirectional LSTM layer
bid_lstm = layers.Bidirectional(
    layers.LSTM(units=100, return_sequences=True,
                recurrent_dropout=0.1))(dropout)

# Add output layer
output = layers.TimeDistributed(
    layers.Dense(num_tags, activation="softmax"))(bid_lstm)

# Define the model
model = Model(input_word, output)  

# Display model summary
model.summary()
model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])
history = model.fit(
    x=X_train,
    y=y_train,
    validation_data=(X_test, y_test),
    batch_size=30, 
    epochs=10,
)
metrics = pd.DataFrame(model.history.history)
metrics.head()

# Plot accuracy
metrics[['accuracy', 'val_accuracy']].plot()
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.show()

# Plot loss
metrics[['loss', 'val_loss']].plot()
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.show()
i = 25
p = model.predict(np.array([X_test[i]]))
p = np.argmax(p, axis=-1)
y_true = y_test[i]

# Display predicted vs true labels
print("Name: LOKESH KUMAR P    Reg No: 212222240054")
print("{:15}{:5}\t {}\n".format("Word", "True", "Pred"))
print("-" * 30)
for w, true, pred in zip(X_test[i], y_true, p[0]):
    print("{:15}{}\t{}".format(words[w-1], tags[true], tags[pred]))

```

## OUTPUT

### Training Loss, Validation Loss Vs Iteration Plot

![Screenshot 2024-11-15 160352](https://github.com/user-attachments/assets/9861cc26-afc2-4265-b01d-d888e9c09786)



### Sample Text Prediction

![Screenshot 2024-11-15 160406](https://github.com/user-attachments/assets/b494f5f0-1717-45fd-9c75-56ff54a6c282)


## RESULT
The experiment results for the Named Entity Recognition (NER) model indicated a training accuracy of 95.4% and a validation accuracy of 92.3%. The training loss was 0.15, and the validation loss was 0.25. These results demonstrate the model's effectiveness in accurately classifying entities in the test dataset.
