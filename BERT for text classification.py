# Connecting gdrive into the google colab

#from google.colab import drive   # Uncomment for google colab

#drive.mount('/content/gdrive')   # Uncomment for google colab

# Uncomment for google colab
#!pip3 install transformers   # To install Huggingface’s transformers library
#!pip3 install torch

# Importing necessary libraries

import re
import os
from collections import Counter
import csv
import numpy as np 
import matplotlib.pyplot as plt 
import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torch.utils.data import RandomSampler
from tqdm import tqdm, trange
from transformers import AdamW, get_linear_schedule_with_warmup 
from transformers import BertConfig, BertForSequenceClassification, BertTokenizer  
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report

PAD_TOKEN_LABEL_ID = CrossEntropyLoss().ignore_index
BATCH_SIZE = 64  # To define a batch size for fine-tuning BERT
LEARNING_RATE_CLASSIFIER = 1e-3
WARMUP_STEPS = 0
LEARNING_RATE_MODEL = 1e-5
GRADIENT_ACCUMULATION_STEPS = 1
MAX_GRAD_NORM = 1.0
SEED = 42
NO_CUDA = False

def read_data(path):
  label_file= open(path) 
  next(label_file)

  dict_labels={}  # this creates a dictionary containing handles and their gender
  gender_stat=[]  # Empty list for genders

  for line in label_file:
      a=line.strip("\n").split("\t")

      dict_labels[a[0]]=a[1]
      gender_stat.append(a[1])
  print(dict_labels)
  return(dict_labels, gender_stat)



path1 = "/fr_gender_info.txt"  #change the path

#read_data(path1)

dict_labels, gender_stat = read_data(path1)

def count(list):
  counter = Counter(gender_stat)
  gender_counter={}
  for k,v in counter.items():
      gender_counter[k]=v
# Counting number of female and male users
  print('male ---> ',gender_counter["M"], "................ female ---> ",gender_counter["F"])
  return(gender_counter)

c=count(gender_stat)

def create_tweets(path):
  tweets=[] # Empty lists to store outputs
  gender=[]

# Download the data in the same directory of the code
  with open(path, 'r') as f: 
    reader = csv.reader(f)
    for row in reader:
      row_split=row[0].split("\t")
      if len( row_split)<9:
        continue

      person_id=row_split[5] # label M or F
      tweet= row_split[8]  # Tweet itself

      if person_id in list(dict_labels):
        tweets.append(tweet)
        gender.append(dict_labels[ person_id])

    size= int(len(tweets )*0.5 )
    tweets=tweets[: size]
    encoder={'M':1,'F':0}
    labels=[]
    for lb in gender:
      labels.append(encoder[lb])
    labels=labels[:size] 
    print(len(tweets))          
    print("tweets and labels are ready")

  return(labels, tweets)


path2= "/tweets-fr.csv"  #change the path
labels, tweets = create_tweets(path2)

#preprocessing the data : removing extra spaces, punctuation 
def parse_text(text):
    text= text.strip().lower() # Lowercase --> we are using uncased model of Bert
    text= text.replace("&nbsp;", " ")
    text= re.sub(r'<br(\s\/)?>', ' ', text)
    text= re.sub(r' +', ' ', text)
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r'^RT[\s]+', '', text)
    text = re.sub(r'^RT[\s]+', '', text)
    text = re.sub(r'https?:\/\/.*[\r\n]*', '', text)
    text= re.sub(r'#', '', text)
    text= re.sub(r'@', '', text)
    text = re.sub('-', ' ', text)
    text = re.sub('<br\s?\/>|<br>', "", text)
    tweet = re.sub(r"(\b|\s+\-?|^\-?)(\d+|\d*\.\d+)\b",'',text)

    return tweet



def read_tweet_data():
    
    data = []
    for tweet in tweets:
        data.append(parse_text(tweet)) 
      
    return data


# Split the data into test and train sets

X= read_tweet_data()  # Specify train set
y=labels              # Specify test set

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.33, random_state=42)


# Because all sentences must have the same length the rest of long sentences is padded with zeros
def padding(arr, n):
    current_len = len(arr)
    if current_len > n:
        return arr[:n]
    extra = n - current_len
    return arr + ([0] * extra)

# Convert tokens into embedding
# Specifing a max lenght for tokens (Bert has a limit for the lenght)
MAX_SENT_LENGTH = 120

def embedding(tokenizer, sentences_with_labels):
    for sentence, label in sentences_with_labels:
        tokens = tokenizer.tokenize(sentence)
        tokens = tokens[:MAX_SENT_LENGTH]
        bert_sent = padding(tokenizer.convert_tokens_to_ids(["CLS"] + tokens + ["SEP"]), n= MAX_SENT_LENGTH)
        # To pad to a max length specified by the max_sent_length
        yield torch.tensor(bert_sent), torch.tensor(label, dtype=torch.int64)  # Convert the integer sequences to tensors

# Take only train set for the training step
def get_data(tokenizer, sampler=RandomSampler, train=True): # Constructing a random sampler for the stored indices : to sampler for sampling the data during training
  
    if train:
       sentences_with_labels = zip(X_train, y_train)
       

    if not train:
       sentences_with_labels = zip(X_test, y_test)

# Creating a list of embedded tokens
    dataset = list(embedding(tokenizer, sentences_with_labels))
    sampler_func = sampler(dataset) if sampler is not None else None
    dataloader = DataLoader(dataset, sampler=sampler_func, batch_size=BATCH_SIZE) # Create dataloaders for both train and test sets
    return dataloader

class Transformers:
    
    model = None

    def __init__(self, tokenizer):

        self.tokenizer = tokenizer
        self.pad_token_label_id = PAD_TOKEN_LABEL_ID
        torch.cuda.empty_cache()  #  Empty the unused memory after processing each batch 
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Function for make predictions
    def predict(self, sentence):
        if self.model is None or self.tokenizer is None:
            self.load()
        embeddings = list(embedding([(sentence, -1)]))
        preds = self._predict_tags_batched(embeddings)
        return preds


# Function for evaluating the model
    def evaluate(self, dataloader):
        
        y_pred = self._predict_tags_batched(dataloader)
        score = classification_report(y_test, y_pred)
        fsc= f1_score(y_test, y_pred, average='micro')

        print('\n')
        print('*' * 40)
        print('Score: ', score)
        print('F1 score: ', fsc)
    
        
    def _predict_tags_batched(self, dataloader):
        preds = []   # Creating an empty list to save the model predictions
        self.model.eval() # To deactivate dropout layer
        for batch in tqdm(dataloader, desc="Computing NER tags"):
            batch = tuple(t.to(self.device) for t in batch)

            with torch.no_grad(): # No gradient calculation.
                outputs = self.model(batch[0])
                _, is_neg = torch.max(outputs[0], 1)  # Returns the maximum value of these output
                preds.extend(is_neg.cpu().detach().numpy())  # Model predictions are stored on GPU. So, push it to CPU

        return preds
# function to train the model
    def train(self, dataloader, model, epochs):
        assert self.model is None 
        model.to(self.device)
        print('our processor ....', self.device)


        self.model = model

        t_total = len(dataloader) // GRADIENT_ACCUMULATION_STEPS * epochs

        # Preparing optimizer and schedule 
        optimizer_grouped_parameters = [
            {"params": model.bert.parameters(), "lr": LEARNING_RATE_MODEL},  # Fine-tune even the pre-trained weights
            {"params": model.classifier.parameters(), "lr": LEARNING_RATE_CLASSIFIER}
        ]
        optimizer = AdamW(optimizer_grouped_parameters) # Optimizing parameters
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=WARMUP_STEPS, num_training_steps=t_total)  # Create a schedule with a learning rate that
            # decreases linearly from the initial lr set in the optimizer to 0,

        global_step = 0   
        tr_loss, logging_loss = 0.0, 0.0
        model.zero_grad() # To clear previously calculated gradients 
        train_iterator = trange(epochs, desc="Epoch")
        self.seeds()
        for _ in train_iterator:
            epoch_iterator = tqdm(dataloader, desc="Iteration")
            for step, batch in enumerate(epoch_iterator):  # To iterate over batches
                model.train()
                batch = tuple(t.to(self.device) for t in batch)
                outputs = model(batch[0], labels=batch[1])
                loss = outputs[0]  

                if GRADIENT_ACCUMULATION_STEPS > 1:
                    loss = loss / GRADIENT_ACCUMULATION_STEPS # Compute loss

                loss.backward() # To backward pass to calculate the gradients

                tr_loss += loss.item() 
                if (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)  # Clips the norm of the overall gradient
                    # It helps in preventing the exploding gradient problem
                    scheduler.step()  # To allow the learning rateto be changed
                    model.zero_grad() #  To make all grads zero
                    global_step += 1
                    optimizer.step() # Performs a parameter update based on the current gradient 

        self.model = model

        return global_step, tr_loss / global_step

    def seeds(self):
        torch.manual_seed(SEED) # To set the seed for generating random numbers
        if self.device == 'gpu':
            torch.cuda.manual_seed_all(SEED)

    def load(self, model_dir='weights/'): # Load weights of best model saved during the training process
        self.tokenizer = BertTokenizer.from_pretrained(model_dir)  # Load the BERT tokenizer
        self.model = BertForSequenceClassification.from_pretrained(model_dir)  
        self.model.to(self.device)

# Select a monolingual model of bert for French  #https://huggingface.co/Geotrend/bert-base-fr-cased
def train(epochs=20, output_dir="weights/"):
    num_labels = 2  # We define the number of labels
# Choose the tokenizer and model
    config = BertConfig.from_pretrained("Geotrend/bert-base-fr-cased", num_labels=num_labels)
    tokenizer = BertTokenizer.from_pretrained('Geotrend/bert-base-fr-cased', do_lower_case=True) # Uncased : we lowercase texts
    model = BertForSequenceClassification.from_pretrained("Geotrend/bert-base-fr-cased", config=config) #Instantiate BERT model

    dataloader = get_data(tokenizer, train=True)
    predictor = Transformers(tokenizer)
    predictor.train(dataloader, model, epochs)

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    
def evaluate(model_dir="weights/"):
    tokenizer = BertTokenizer.from_pretrained('Geotrend/bert-base-fr-cased', do_lower_case=True)
    dataloader = get_data(tokenizer, train=False, sampler=None)
    predictor = Transformers(tokenizer)
    predictor.load(model_dir=model_dir)
    predictor.evaluate(dataloader)


path = '/content/gdrive/My Drive/weights/'

train(epochs=5, output_dir=path)
evaluate(model_dir=path)
