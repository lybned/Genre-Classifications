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


		
class DistillBertUtil:
	def __init__():
		pass
		
	def getTokenizer():
		tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
		return tokenizer
		
	def getDistillBertData(texts, genres, tokenizer):
		#tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
		print(tokenizer)
		# Create a Dataset
		data = {'text': texts, 'label': genres}
		dataset = Dataset.from_dict(data)


		# Encode the labels
		labels = ClassLabel(names=list(set(genres)))
		dataset = dataset.map(lambda examples: {'label': labels.str2int(examples['label'])})
		
		def tokenize_function(examples):
			return tokenizer(examples['text'], truncation=True)
			
		tokenized_datasets = dataset.map(tokenize_function, batched=True)
		
		tokenized_datasets = tokenized_datasets.train_test_split(test_size=0.2)
		train_dataset = tokenized_datasets['train']
		test_dataset = tokenized_datasets['test']
		
		return train_dataset, test_dataset
		
	def getDistillBert(labels):
		model_2 = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=labels)
		return model_2
		
	def trainModel(model,train_dataset, test_dataset, tokenizer):
		# Training arguments
		training_args = TrainingArguments(
			output_dir='./results',
			evaluation_strategy="epoch",
			learning_rate=2e-5,
			per_device_train_batch_size=64,
			per_device_eval_batch_size=64,
			num_train_epochs=3,
			weight_decay=0.01,
		)

		# Trainer
		data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
		trainer = Trainer(
			model=model,
			args=training_args,
			train_dataset=train_dataset,
			eval_dataset=test_dataset,
			tokenizer=tokenizer,
			data_collator=data_collator,
		)

		# Train the model
		trainer.train()
		