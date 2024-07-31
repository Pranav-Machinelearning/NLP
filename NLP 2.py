import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
import seaborn as sns
import matplotlib.pyplot as plt
from nltk.stem import WordNetLemmatizer

# Load data
semantic_data = pd.read_csv(r"C:\Users\prana\OneDrive\Documents\Dataset\Semantic analysis.csv", names=["ID", "Hashtags", "Message status", "Twitter Messages"])
semantic_data['Message status'] = semantic_data['Message status'].map({'negative': 0, 'positive': 1})

# Oversample the minority class to balance the dataset
no_r = semantic_data[semantic_data['Message status'] == 0]
yes_r = semantic_data[semantic_data['Message status'] == 1]
re = resample(yes_r, replace=True, n_samples=len(no_r), random_state=123)
overspl = pd.concat([no_r, re])

# Preprocess text data
emojis = {':)': 'smile', ':-)': 'smile', ';d': 'wink', ':-E': 'vampire', ':(': 'sad', 
          ':-(': 'sad', ':-<': 'sad', ':P': 'raspberry', ':O': 'surprised',
          ':-@': 'shocked', ':@': 'shocked',':-$': 'confused', ':\\': 'annoyed', 
          ':#': 'mute', ':X': 'mute', ':^)': 'smile', ':-&': 'confused', '$_$': 'greedy',
          '@@': 'eyeroll', ':-!': 'confused', ':-D': 'smile', ':-0': 'yell', 'O.o': 'confused',
          '<(-_-)>': 'robot', 'd[-_-]b': 'dj', ":'-)": 'sadsmile', ';)': 'wink', 
          ';-)': 'wink', 'O:-)': 'angel','O*-)': 'angel','(:-D': 'gossip', '=^.^=': 'cat'}

stopwordlist = ['a', 'about', 'above', 'after', 'again', 'ain', 'all', 'am', 'an',
                'and','any','are', 'as', 'at', 'be', 'because', 'been', 'before',
                'being', 'below', 'between','both', 'by', 'can', 'd', 'did', 'do',
                'does', 'doing', 'down', 'during', 'each','few', 'for', 'from', 
                'further', 'had', 'has', 'have', 'having', 'he', 'her', 'here',
                'hers', 'herself', 'him', 'himself', 'his', 'how', 'i', 'if', 'in',
                'into','is', 'it', 'its', 'itself', 'just', 'll', 'm', 'ma',
                'me', 'more', 'most','my', 'myself', 'now', 'o', 'of', 'on', 'once',
                'only', 'or', 'other', 'our', 'ours','ourselves', 'out', 'own', 're',
                's', 'same', 'she', "shes", 'should', "shouldve",'so', 'some', 'such',
                't', 'than', 'that', "thatll", 'the', 'their', 'theirs', 'them',
                'themselves', 'then', 'there', 'these', 'they', 'this', 'those', 
                'through', 'to', 'too','under', 'until', 'up', 've', 'very', 'was',
                'we', 'were', 'what', 'when', 'where','which','while', 'who', 'whom',
                'why', 'will', 'with', 'won', 'y', 'you', "youd","youll", "youre",
                "youve", 'your', 'yours', 'yourself', 'yourselves']

def preprocesstweet(textdata):
    tweettext = []
    wordLemm = WordNetLemmatizer()
    urlPattern = r"((http://)[^ ]*|(https://)[^ ]*|( www\.)[^ ]*)"
    userPattern = '@[^\s]+'
    alphaPattern = "[^a-zA-Z0-9]"
    sequencePattern = r"(.)\1\1+"
    seqReplacePattern = r"\1\1"
    
    for tweet in textdata:
        tweet = tweet.lower()
        tweet = re.sub(urlPattern,' URL',tweet)
        for emoji in emojis.keys():
            tweet = tweet.replace(emoji, "EMOJI" + emojis[emoji])        
        tweet = re.sub(userPattern,' USER', tweet)        
        tweet = re.sub(alphaPattern, " ", tweet)
        tweet = re.sub(sequencePattern, seqReplacePattern, tweet)

        tweetwords = ''
        for word in tweet.split():
            if len(word) > 1:
                word = wordLemm.lemmatize(word)
                tweetwords += (word+' ')
            
        tweettext.append(tweetwords)
        
    return tweettext

text = overspl['Twitter Messages'].tolist()
tweettext = preprocesstweet(text)

# Initialize the TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer(max_features=1000)
tfidf_features = tfidf_vectorizer.fit_transform(tweettext).toarray()

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(tfidf_features, overspl['Message status'], test_size=0.2, random_state=42)

# CNN Model
cnn_model = models.Sequential([
    layers.Reshape((1000, 1), input_shape=(1000,)),  # Reshape to add a single channel
    layers.Conv1D(32, 3, activation='relu'),
    layers.MaxPooling1D(2),
    layers.Conv1D(64, 3, activation='relu'),
    layers.MaxPooling1D(2),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

cnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
cnn_model.fit(X_train, y_train, epochs=5, batch_size=64, validation_split=0.2)
cnn_loss, cnn_acc = cnn_model.evaluate(X_test, y_test)
print(f"CNN Test Accuracy: {cnn_acc}")

# Calculate Confusion Matrix for CNN
cnn_predictions = cnn_model.predict(X_test)
cnn_predictions = (cnn_predictions > 0.5).astype(int).flatten()
cnn_cm = confusion_matrix(y_test, cnn_predictions)
print("CNN Confusion Matrix:")
print(cnn_cm)
sns.heatmap(cnn_cm, annot=True, fmt='d', cmap='Blues')
plt.title('CNN Confusion Matrix')
plt.show()

# Calculate Precision, Recall, and F1 Score for CNN
cnn_precision = precision_score(y_test, cnn_predictions)
cnn_recall = recall_score(y_test, cnn_predictions)
cnn_f1 = f1_score(y_test, cnn_predictions)
print(f"CNN Precision: {cnn_precision}")
print(f"CNN Recall: {cnn_recall}")
print(f"CNN F1 Score: {cnn_f1}")

# RNN Model
rnn_model = models.Sequential([
    layers.Embedding(input_dim=1000, output_dim=128, input_length=1000),
    layers.SimpleRNN(128),
    layers.Dense(1, activation='sigmoid')
])

rnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
rnn_model.fit(X_train, y_train, epochs=5, batch_size=64, validation_split=0.2)
rnn_loss, rnn_acc = rnn_model.evaluate(X_test, y_test)
print(f"RNN Test Accuracy: {rnn_acc}")

# Calculate Confusion Matrix for RNN
rnn_predictions = rnn_model.predict(X_test)
rnn_predictions = (rnn_predictions > 0.5).astype(int).flatten()
rnn_cm = confusion_matrix(y_test, rnn_predictions)
print("RNN Confusion Matrix:")
print(rnn_cm)
sns.heatmap(rnn_cm, annot=True, fmt='d', cmap='Blues')
plt.title('RNN Confusion Matrix')
plt.show()

# Calculate Precision, Recall, and F1 Score for RNN
rnn_precision = precision_score(y_test, rnn_predictions)
rnn_recall = recall_score(y_test, rnn_predictions)
rnn_f1 = f1_score(y_test, rnn_predictions)
print(f"RNN Precision: {rnn_precision}")
print(f"RNN Recall: {rnn_recall}")
print(f"RNN F1 Score: {rnn_f1}")

# LSTM Model
lstm_model = models.Sequential([
    layers.Embedding(input_dim=1000, output_dim=128, input_length=1000),
    layers.LSTM(128),
    layers.Dense(1, activation='sigmoid')
])

lstm_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
lstm_model.fit(X_train, y_train, epochs=5, batch_size=64, validation_split=0.2)
lstm_loss, lstm_acc = lstm_model.evaluate(X_test, y_test)
print(f"LSTM Test Accuracy: {lstm_acc}")

# Calculate Confusion Matrix for LSTM
lstm_predictions = lstm_model.predict(X_test)
lstm_predictions = (lstm_predictions > 0.5).astype(int).flatten()
lstm_cm = confusion_matrix(y_test, lstm_predictions)
print("LSTM Confusion Matrix:")
print(lstm_cm)
sns.heatmap(lstm_cm, annot=True, fmt='d', cmap='Blues')
plt.title('LSTM Confusion Matrix')
plt.show()

# Calculate Precision, Recall, and F1 Score for LSTM
lstm_precision = precision_score(y_test, lstm_predictions)
lstm_recall = recall_score(y_test, lstm_predictions)
lstm_f1 = f1_score(y_test, lstm_predictions)
print(f"LSTM Precision: {lstm_precision}")
print(f"LSTM Recall: {lstm_recall}")
print(f"LSTM F1 Score: {lstm_f1}")

# Transformer Model
from tensorflow.keras.layers import TextVectorization

def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    x = layers.LayerNormalization(epsilon=1e-6)(inputs)
    x = layers.MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(x, x)
    x = layers.Dropout(dropout)(x)
    res = x + inputs

    x = layers.LayerNormalization(epsilon=1e-6)(res)
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    return x + res

inputs = layers.Input(shape=(1000,))
embedding_layer = layers.Embedding(input_dim=1000, output_dim=128)(inputs)
transformer_block = transformer_encoder(embedding_layer, head_size=128, num_heads=4, ff_dim=128)
x = layers.GlobalAveragePooling1D()(transformer_block)
x = layers.Dropout(0.1)(x)
x = layers.Dense(64, activation="relu")(x)
x = layers.Dropout(0.1)(x)
outputs = layers.Dense(1, activation="sigmoid")(x)

transformer_model = models.Model(inputs, outputs)
transformer_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
transformer_model.fit(X_train, y_train, epochs=5, batch_size=64, validation_split=0.2)
transformer_loss, transformer_acc = transformer_model.evaluate(X_test, y_test)
print(f"Transformer Test Accuracy: {transformer_acc}")

# Calculate Confusion Matrix for Transformer
transformer_predictions = transformer_model.predict(X_test)
transformer_predictions = (transformer_predictions > 0.5).astype(int).flatten()
transformer_cm = confusion_matrix(y_test, transformer_predictions)
print("Transformer Confusion Matrix:")
print(transformer_cm)
sns.heatmap(transformer_cm, annot=True, fmt='d', cmap='Blues')
plt.title('Transformer Confusion Matrix')
plt.show()

# Calculate Precision, Recall, and F1 Score for Transformer
transformer_precision = precision_score(y_test, transformer_predictions)
transformer_recall = recall_score(y_test, transformer_predictions)
transformer_f1 = f1_score(y_test, transformer_predictions)
print(f"Transformer Precision: {transformer_precision}")
print(f"Transformer Recall: {transformer_recall}")
print(f"Transformer F1 Score: {transformer_f1}")
