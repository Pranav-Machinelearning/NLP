import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt
from sklearn.utils import resample
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns

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
#         tweet = tweet.lower()
#         tweet = re.sub(urlPattern,' URL',tweet)
#         for emoji in emojis.keys():
#             tweet = tweet.replace(emoji, "EMOJI" + emojis[emoji])        
#         tweet = re.sub(userPattern,' USER', tweet)        
#         tweet = re.sub(alphaPattern, " ", tweet)
#         tweet = re.sub(sequencePattern, seqReplacePattern, tweet)

        tweetwords = ''
        for word in tweet.split():
            if word not in stopwordlist and len(word) > 1:
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

# Function to evaluate and collect metrics for each model
def evaluate_model(model, X_test, y_test):
    y_pred = (model.predict(X_test) > 0.5).astype("int32")
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    return accuracy, precision, recall, f1, cm

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
cnn_metrics = evaluate_model(cnn_model, X_test, y_test)

# RNN Model
rnn_model = models.Sequential([
    layers.Embedding(input_dim=1000, output_dim=128, input_length=1000),
    layers.SimpleRNN(128),
    layers.Dense(1, activation='sigmoid')
])

rnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
rnn_model.fit(X_train, y_train, epochs=5, batch_size=64, validation_split=0.2)
rnn_metrics = evaluate_model(rnn_model, X_test, y_test)

# LSTM Model
lstm_model = models.Sequential([
    layers.Embedding(input_dim=1000, output_dim=128, input_length=1000),
    layers.LSTM(128),
    layers.Dense(1, activation='sigmoid')
])

lstm_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
lstm_model.fit(X_train, y_train, epochs=5, batch_size=64, validation_split=0.2)
lstm_metrics = evaluate_model(lstm_model, X_test, y_test)

# Transformer Model
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
transformer_metrics = evaluate_model(transformer_model, X_test, y_test)

# Compare Metrics
models_metrics = {
    'Model': ['CNN', 'RNN', 'LSTM', 'Transformer'],
    'Accuracy': [cnn_metrics[0], rnn_metrics[0], lstm_metrics[0], transformer_metrics[0]],
    'Precision': [cnn_metrics[1], rnn_metrics[1], lstm_metrics[1], transformer_metrics[1]],
    'Recall': [cnn_metrics[2], rnn_metrics[2], lstm_metrics[2], transformer_metrics[2]],
    'F1 Score': [cnn_metrics[3], rnn_metrics[3], lstm_metrics[3], transformer_metrics[3]]
}

metrics_df = pd.DataFrame(models_metrics)

# Plotting
plt.figure(figsize=(12, 8))
metrics_df.set_index('Model').plot(kind='bar')
plt.title('Comparison of Model Performance Metrics')
plt.xlabel('Model')
plt.ylabel('Score')
plt.ylim(0, 1)
plt.legend(loc='best')
plt.show()

# Display Confusion Matrices
fig, axs = plt.subplots(2, 2, figsize=(12, 12))

sns.heatmap(cnn_metrics[4], annot=True, fmt='d', cmap='Blues', ax=axs[0, 0])
axs[0, 0].set_title('CNN Confusion Matrix')

sns.heatmap(rnn_metrics[4], annot=True, fmt='d', cmap='Blues', ax=axs[0, 1])
axs[0, 1].set_title('RNN Confusion Matrix')

sns.heatmap(lstm_metrics[4], annot=True, fmt='d', cmap='Blues', ax=axs[1, 0])
axs[1, 0].set_title('LSTM Confusion Matrix')

sns.heatmap(transformer_metrics[4], annot=True, fmt='d', cmap='Blues', ax=axs[1, 1])
axs[1, 1].set_title('Transformer Confusion Matrix')

plt.show()
