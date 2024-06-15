from src.util import *
from src.config import *

# main.py
#import sys
#import os

# Add the src directory to the system path
#sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.DistillBert.util import *
from src.RandomForest.util import *
from src.TFIDF.util import *
from src.Word2Vec.util import *
from src.Recommender.util import *

def randomForestTrainTest(plotList, genreList):
	
	X_train, X_test, y_train, y_test = DataUtil.dataSplit(plotList, genreList, splitRatio)
	
	#print("Random Forest Models with TF-IDF")
	randomForestModel = RandomForestUtil.getRandomForestModel(rfNodes)
	randomForestModel = RandomForestUtil.fitModel(randomForestModel, X_train, y_train)
	
	f1 = ResultUtil.getF1Score(randomForestModel, X_test, y_test)
	print("Weighted F1-Score on Test Set (First):", f1)
	
	cvResult = ResultUtil.crossValidation(randomForestModel, X_train, y_train, cv)
	print("Cross-validated scores:", cvResult)
	print("Average F1-Weighted Score:", cvResult.mean())
	
	gridSearchResult = ResultUtil.gridSearch(randomForestModel, cv, X_train, y_train)
	# Best parameters and best score
	print("Best Parameters:", gridSearchResult.best_params_)
	print("Best Weighted F1-Score from Grid Search:", gridSearchResult.best_score_)	
	return randomForestModel

if __name__ == "__main__":
	print("Running Main")
	d = DataUtil.read_data(data_dir)
	

	# Get the returned data
	filtered_df = DataUtil.getPlotsGenre(d,genre_List)
	plotList = filtered_df['Plot_Clean']
	genreList = filtered_df['Genre']
	'''
	print("Finished Reading Data")	
	
	
	print("Random Forest Models with TF-IDF")		
	plotListEncoded = TFIDFEncoding.encodeData(plotList)
	modelTFIDF = randomForestTrainTest(plotListEncoded, genreList)

	print("Random Forest Models with Word2Vec")	
	Word2VecModel, tokenizedList = Word2VecEncoding.trainWord2VecModel(plotList)
	plotListWord2Vec = Word2VecEncoding.Word2VecEncode(tokenizedList,Word2VecModel)
	modelWord2Vec = randomForestTrainTest(plotListWord2Vec,genreList)

	tokenizer = DistillBertUtil.getTokenizer()
	trainData, testData = DistillBertUtil.getDistillBertData(plotList, genreList, tokenizer)
	bertModel = DistillBertUtil.getDistillBert(3)
	print("Start Training")
	DistillBertUtil.trainModel(bertModel, trainData, testData, tokenizer)
	'''
	matrix = RecommenderSystem.getRecommenderMatrix(filtered_df)
	movieIndex = int(input("Enter the index of the movie in the dataset: "))
	result = recommend_movies(movieIndex, matrix, filtered_df, num_recommendations=5)
	print(result)