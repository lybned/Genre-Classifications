import nltk
from nltk.probability import FreqDist
from nltk.tokenize import word_tokenize
from nltk.chunk import ne_chunk
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from nltk import pos_tag, RegexpParser
from nltk.tokenize import sent_tokenize
import string
from nltk.corpus import stopwords
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix
import torch
from transformers import GPT2Tokenizer, GPT2ForSequenceClassification
from transformers import Trainer, TrainingArguments
import pandas as pd
import numpy as np
from torch import nn, optim
from transformers import GPT2Model, GPT2Tokenizer
from torch.optim import Adam
from tqdm import tqdm
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import f1_score

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import OneHotEncoder

from collections import Counter
from transformers import DistilBertModel, DistilBertTokenizer


from sklearn.preprocessing import MultiLabelBinarizer

def get_embedding(text, tokenizer, model):
    # Tokenize the input text
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    
    # Generate embeddings
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Use the mean of the token embeddings as the sentence embedding
    embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings.squeeze().numpy()
	
# Function to create binary vector for keyword presence
def count_keywords(text, keywords):
	# Convert the text to lower case for case insensitive matching
	text = text.lower()
	# Initialize a dictionary to store the count of each keyword
	keyword_count = {keyword: 0 for keyword in keywords}
	
	# Tokenize the text using NLTK
	words = word_tokenize(text)
	
	# Iterate through each word in the tokenized text
	for word in words:
		# If the word is in the keywords list, increment its count
		if word in keyword_count:
			keyword_count[word] += 1
			
	# Create a list of counts corresponding to the keywords
	keyword_counts_list = [keyword_count[keyword] for keyword in keywords]
	
	return keyword_counts_list
	
def recommend_movies(title, cosine_sim, df, num_recommendations=5):
    # Get the index of the movie that matches the title
    idx = title#df.index[df['Title'] == title].tolist()
    print(df.iloc[idx])
    #print(idx)
    # Get the pairwise similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))
    
    print(sim_scores)
    
    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Get the scores of the 10 most similar movies
    sim_scores = sim_scores[1:num_recommendations+1]
    
    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]
    
    # Return the top 10 most similar movies
    return (df.iloc[movie_indices], sim_scores)
	
class RecommenderSystem:
	def __init__():
		pass
		
	def getRecommenderMatrix(filtered_df):
	
		# Load pre-trained DistilBERT model and tokenizer
		tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
		model = DistilBertModel.from_pretrained('distilbert-base-uncased')
	
		filtered_df['Plot_Embedding'] = filtered_df['Plot_Clean'].apply(lambda plot: get_embedding(plot, tokenizer, model))


		# Define the keywords to check in the plots
		keywords = ["police", "men", "death", "kills", "life", "home", "family", "wife", "son", "house"] 




		# Stack plot embeddings into a matrix
		plot_embeddings = np.stack(filtered_df['Plot_Embedding'].values)

		# Compute cosine similarity matrix
		cosine_sim = cosine_similarity(plot_embeddings)

		filtered_df['Keyword_Vector'] = filtered_df['Plot_Clean'].apply(lambda plot: count_keywords(plot, keywords))
		keyword_vectors = np.stack(filtered_df['Keyword_Vector'].values)

		# One-hot encode the genres
		mlb = MultiLabelBinarizer()
		genre_vectors = mlb.fit_transform(filtered_df['Genre'])

		# Normalize genre vectors
		genre_vectors = genre_vectors / np.linalg.norm(genre_vectors, axis=1, keepdims=True)


		# One-hot encode the origins
		origin_encoder = OneHotEncoder(sparse=False)
		origin_vectors = origin_encoder.fit_transform(filtered_df[['Origin/Ethnicity']])

		# Normalize origin vectors
		origin_vectors = origin_vectors / np.linalg.norm(origin_vectors, axis=1, keepdims=True)

		# Combine genre vectors and plot embeddings
		combined_vectors = np.hstack((genre_vectors,origin_vectors, plot_embeddings, keyword_vectors))

		# Compute combined cosine similarity matrix
		combined_cosine_sim = cosine_similarity(combined_vectors)

		return combined_cosine_sim
		