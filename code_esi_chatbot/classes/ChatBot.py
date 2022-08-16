import json
import random
import string
import nltk
from nltk.stem import WordNetLemmatizer
import numpy as np
from tensorflow.python.keras.models import load_model
import os

nltk.download('punkt')
nltk.download('wordnet')


# ------ Constants ------
lemmatizer = WordNetLemmatizer()
MODEL_FILE = os.path.join(os.path.dirname(__file__), './model.h5')
INTENTS_FILE = os.path.join(os.path.dirname(__file__), './intents.json')
DONT_UNDERSTAND_RESPONSE = "I'm sorry, can you please rephrase your question?"
model = load_model(MODEL_FILE)
intents_json = json.load(open(INTENTS_FILE))
# ------ Static Functions ------

words = [] # list of all words used in the training data
classes = [] # list of all classes used in the training data, i.e. the intents
for intent in intents_json['intents']:
    for pattern in intent['patterns']:
        # tokenize each word in the pattern
        tokenized_pattern = nltk.word_tokenize(pattern)
        words.extend(tokenized_pattern)
    # add the tag to the class list even there's no pattern in the intent
    if intent['tag'] not in classes:
        classes.append(intent['tag'])
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in string.punctuation]
words = sorted(list(set(words)))
classes = sorted(list(set(classes)))

# ------ ChatBot Class ------
class ChatBot:
   def __init__(self):
        self.intents_json = intents_json
        self.result = DONT_UNDERSTAND_RESPONSE
        self.model = model
        self.vocab = words
        self.labels = classes
    
   def clean_input(self):
        tokens = nltk.word_tokenize(self.user_input)
        tokens = [lemmatizer.lemmatize(w.lower()) for w in tokens]
        self.tokens = tokens
    
   def bag_of_words(self):
        bow = [0] * len(self.vocab)
        for w in self.tokens:
            for index, word in enumerate(self.vocab):
                if word == w:
                    bow[index] = 1
        self.bow = np.array(bow)
    
   def pred_class(self):
        result = self.model.predict(np.array([self.bow]))[0]
        thresh = 0.5
        y_pred = [[index, res] for index, res in enumerate(result) if res > thresh ]
        y_pred.sort(key=lambda x: x[1], reverse=True)
        results_list = [self.labels[i[0]] for i in y_pred]
        self.pred_list = results_list
    
   def get_response(self):
        self.result = DONT_UNDERSTAND_RESPONSE
        if len(self.pred_list) > 0:
            tag = self.pred_list[0]
            list_intents = self.intents_json['intents']
            for intent in list_intents:
                if intent['tag'] == tag:
                    self.result = random.choice(intent['responses'])
                    break
    
   def get_result(self, user_input):
        self.user_input = user_input
        self.clean_input()
        self.bag_of_words()
        self.pred_class()
        self.get_response()
        return self.result

