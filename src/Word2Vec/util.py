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



def doc_vectorizer(tokens, model):
    vectors = [model.wv[token] for token in tokens if token in model.wv]

    return sum(vectors) / len(vectors) if vectors else None



	
class Word2VecEncoding:
	def __init__():
		pass
		
	def trainWord2VecModel(textList):
		textListToken = [word_tokenize(text) for text in textList]
		word2vec_model = Word2Vec(textListToken, window=5, min_count=1, workers=4)
		return word2vec_model, textListToken
		
	def Word2VecEncode(TokenList, model):
		trainingData = [doc_vectorizer(tokens, model) for tokens in TokenList]
		return trainingData
		
