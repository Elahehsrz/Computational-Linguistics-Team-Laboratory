# Importing necessary libraries
import csv
import numpy as np
from collections import Counter
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.linear_model import Perceptron
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import warnings
warnings.filterwarnings('ignore')
from sklearn import utils
from nltk.tokenize import word_tokenize
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
stopwords_list = stopwords.words('french') 
import string
nltk.download('punkt')


# Function to read the data and create a dictionary contatining handles and geners
def read_gender_data(path):
  label_file= open(path)
  next(label_file)

  dict_labels = {}  #  Initiate an empty dictionary containing handles and their gender

  for line in label_file:  # Read line by line
      a = line.strip("\n").split("\t")    # split words
      dict_labels[a[0]] = a[1]

  return dict_labels

"""First we get the genders of the tweeters"""

path1 = "/fr_gender_info.txt"   # change the path
genders = read_gender_data(path1)  # Use the read_gender_data function
print(genders)


# Function to count genders
def count(genders):
  counter = Counter(genders)
  gender_counter = {}
  for k, v in counter.items():
      gender_counter[k] = v
  return gender_counter

print(count(genders.values())) 

def create_tweets(path, ids, max_len=-1):
  tweets = []  

  # Download the data in the same directory of the code
  with open(path, 'r') as f: # The path to the tweets file
    reader = csv.reader(f)

    for row in reader:
      row_split = row[0].split("\t")

      if len(row_split) >= 9: # ignore short tweets
        person_id = row_split[5] # the id of the tweeter
        tweet = row_split[8]  # Tweet itself

        if person_id in ids:
          tweets.append((person_id, tweet))

        if max_len != -1 and len(tweets) >= max_len:  # We set a max lenght to ignore tweets more than this number
          break

  return tweets

path2 = "/tweets-fr.csv"
tweets = create_tweets(path2, genders.keys(), 2000)   # We set the lenght to 2000
print(tweets)
print(len(tweets))


# Function to preprocess our tweets
def clean_tweets(tweets, min_length=4):
    output = []

    for id, tweet in tweets:
        words = tweet.split()
        clean_words = [] # Initiate an empty list for tweets

        for w in words:  # To remove extra punctuation
            if (w == ""):
                continue
            elif w.startswith('@'):
                continue
            elif w.startswith('#'):
                continue
            elif w.startswith('http'):    
                continue
            else:
                w = w.strip('0123456789') # strip digits
                w = w.strip('\'"!?$,.:;/-_<>@)(*^%][—♡❤️#')  # strip punctuation
                clean_words.append(w.lower())
  
        if len(clean_words) >= min_length:  # To remove short tweets
            tweet_clean = ' '.join(clean_words)
            output.append((id, tweet_clean))

    return output

tweets = clean_tweets(tweets)  
print(tweets)

gender_map = {
    'M': 0,
    'F': 1,
}


# Function to create the data using tweets and corresponding gender ( based on their handles)
def create_data(genders, tweets):
  inputs = []
  targets = []

  for id, tweet in tweets:
    inputs.append(tweet)
    gender = genders[id]
    targets.append(gender)

  return inputs, targets

texts, labels = create_data(genders, tweets)
print(texts)
print(labels)

# Tweet vectorizing by doc2vec model
def vectorize(texts, labels, path):
  tagged_tweets = []

  for text, label in zip(texts, labels):
    tagged_tweets.append(TaggedDocument(words=word_tokenize(text), tags=label))  # Tokens and ids

  #  tokenize text using NLTK tokenizer
  model_dm = Doc2Vec(dm=1, vector_size=200, negative=5, workers=5, hs=0, min_count=2, alpha=0.025, min_alpha=0.0001)
  # dm=1 means ‘distributed memory’ (PV-DM). Distributed Memory model preserves the word order in a document 
  # [https://medium.com/@mishra.thedeepak/doc2vec-simple-implementation-example-df2afbbfbad5]
  model_dm.build_vocab(tagged_tweets)
  train_tweets  = utils.shuffle(tagged_tweets)  # Shuffle arrays or sparse matrices in a consistent way.
  model_dm.train(tagged_tweets,total_examples=len(tagged_tweets), epochs=100) # To train the model
  model_dm.save(path)
  return model_dm, tagged_tweets

# Use the vectorize function
path3 = './tweetModel.d2v'
model_dm, tagged_tweets = vectorize(texts, labels, path3)

def prepare_data(model_dm, tagged_tweets):
  feature_vectors=[] 
  targets=[]

  for tw in tagged_tweets:
      vec = model_dm.infer_vector(tw.words, steps=10) # To infer vector for a new document
      gender=tw.tags[0] # Stores labels
      feature_vectors.append(vec) # To obtain feature vectors
      targets.append(gender)
      
  # Split the data into test and train sets, specify train and test sets
  X_train, X_test, y_train, y_test = train_test_split(feature_vectors, targets, test_size=0.25, random_state=42)
  return X_train, X_test, y_train, y_test


X_train, X_test, y_train, y_test = prepare_data(model_dm, tagged_tweets)

# Training the logistic regression classifier
def train_logistic_regressor(X_train, y_train):
  lg = LogisticRegression(C=1)
  lg.fit(X_train, y_train)
  return lg

# Testing
def test_logistic_regressor(lg, X_test, y_test):
  y_pred = lg.predict(X_test)
  score = classification_report(y_test, y_pred)
  print('LG F1 score (weighted): {}'.format(f1_score(y_test, y_pred, average='weighted')))
  print('LG F1 score (micro): {}'.format(f1_score(y_test, y_pred, average='micro')))
  print('LG score', score)

lg = train_logistic_regressor(X_train, y_train)
test_logistic_regressor(lg, X_test, y_test)


# Training the support vector machine classifier
def train_svm(X_train, y_train):
  svm = LinearSVC()
  svm.fit(X_train, y_train)
  return svm


# Testing
def test_svm(svm, X_test, y_test):
  y_pred = svm.predict(X_test)
  score = classification_report(y_test, y_pred)
  print('SVM F1 score (weighted): {}'.format(f1_score(y_test, y_pred, average='weighted')))
  print('SVM F1 score (micro): {}'.format(f1_score(y_test, y_pred, average='micro')))
  print('SVM score' , score)

svm = train_svm(X_train, y_train)
test_svm(svm, X_test, y_test)


# Training the Perceptron classifier
def train_perceptron(X_train, y_train):
  per = Perceptron()
  per.fit(X_train, y_train)
  return per


# Testing
def test_perceptron(per, X_test, y_test):
  y_pred = per.predict(X_test)
  score = classification_report(y_test, y_pred)
  print('Perceptron F1 score (weighted): {}'.format(f1_score(y_test, y_pred, average='weighted')))
  print('Perceptron F1 score (micro): {}'.format(f1_score(y_test, y_pred, average='micro')))
  print('Perceptron score', score)

per = train_perceptron(X_train, y_train)
test_perceptron(per, X_test, y_test)
