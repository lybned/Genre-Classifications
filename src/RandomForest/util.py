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
from transformers import Trainer, TrainingArguments
from sklearn.metrics import accuracy_score
import numpy as np
from torch import nn, optim
from torch.optim import Adam
from tqdm import tqdm
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import f1_score
import seaborn as sns
from datasets import Dataset, ClassLabel
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding
from transformers import BertTokenizer, BertForMaskedLM, BertForSequenceClassification


			
class RandomForestUtil:
	def __init__():
		pass
		
	def getRandomForestModel(n):
		return RandomForestClassifier(n_estimators=n, random_state=42)
	  
	  
	def fitModel(model,data,label):
		model.fit(data, label)
		return model


		