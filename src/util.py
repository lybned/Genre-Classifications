import pandas as pd
from nltk.corpus import stopwords
import os
import nltk
from nltk.probability import FreqDist
from nltk.tokenize import word_tokenize
from nltk.chunk import ne_chunk
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from nltk import pos_tag, RegexpParser
from nltk.tokenize import sent_tokenize
import pandas as pd
import string
from nltk.corpus import stopwords
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix
from gensim.models import Word2Vec
from gensim.models import FastText
import gensim.downloader
from transformers import GPT2Tokenizer, GPT2ForSequenceClassification, GPT2Model
from transformers import Trainer, TrainingArguments
from sklearn.metrics import accuracy_score
import numpy as np
from torch import nn, optim
from torch.optim import Adam
from tqdm import tqdm
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import f1_score
import seaborn as sns

def doc_vectorizer(tokens, model):
    vectors = [model.wv[token] for token in tokens if token in model.wv]

    return sum(vectors) / len(vectors) if vectors else None

def remove_punctuation(text):
    # Define punctuation
    punctuation = string.punctuation
    # Remove punctuation from text
    return ''.join(char for char in text if char not in punctuation)

# Define a function to remove stop words
def remove_stopwords(text):
	stopwordSet = set(stopwords.words('english'))
    words = text.split()  # Split text into words
    filtered_words = [word for word in words if word.lower() not in stopwordSet]
    return ' '.join(filtered_words)

class DataUtil:
	def __init__(self):
		pass
		
	def read_data(self, filepath):
		self.dataset = pd.read_csv(filepath)
		
	def getPlotsGenre(self, genreList):
		filtered_df = df[df['Genre'].isin(genreList)]
		
		# Apply the function to the DataFrame column
		filtered_df['Plot_Clean'] = filtered_df['Plot'].apply(remove_punctuation)
		
		# Apply the function to the DataFrame column
		filtered_df['Plot_Clean'] = filtered_df['Plot_Clean'].apply(remove_stopwords)
		
		return filtered_df['Plot_Clean'], filtered_df['Genre']
		
	def dataSplit(self, dataList, labelList, splitRatio):
		return train_test_split(dataList, labelList, test_size=splitRatio, random_state=42)
			
class RandomForestUtil:
	def __init__():
		pass
		
	def getRandomForestModel(self, n):
		return RandomForestClassifier(n_estimators=n, random_state=42)
	  
	  
	def fitModel(self,model,data,label):
        model.fit(data, labels)
        return model
	

class ResultUtil:
	def __init__():
		pass
		
	def getF1Score(self, model, X_test,y_test):
		# Evaluate the model
		y_pred = rf_model.predict(X_test)

		# Calculate weighted F1-score
		f1 = f1_score(y_test, y_pred, average='weighted')
		return f1
		
	def crossValidation(self, model, X_train, y_train, cv):
		return cross_val_score(model, X_train, y_train, cv=cv, scoring='f1_weighted')
		
	def gridSearch(self, model, cv, X_train, y_train):
		# Define parameter grid for GridSearch
		param_grid = {
		  'n_estimators': [100, 200, 300],
		  'max_depth': [None, 10, 20, 30],
		  'min_samples_split': [2, 5, 10],
		  'min_samples_leaf': [1, 2, 4]
		}
		
		# Create a GridSearchCV object
		grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, 
								cv=5, scoring='f1_weighted', n_jobs=-1)

		# Fit the grid search to the data
		grid_search.fit(X_train, y_train)
		
		return grid_search
	
class Word2VecEncoding:
	def __init__():
		pass
		
	def trainWord2VecModel(self,textList):
		textListToken = [word_tokenize(text) for text in textList]
		word2vec_model = Word2Vec(textListToken, window=5, min_count=1, workers=4)
		return word2vec_model
		
	def Word2VecEncode(self,TokenList, model):
		trainingData = [doc_vectorizer(tokens, model) for tokens in TokenList]
		return trainingData