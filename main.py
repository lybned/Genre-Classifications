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

if __name__ == "__main__":
	print("Running Main")
	d = DataUtil.read_data(data_dir)
	
	#print("Finished Reading Data")

	plotList, genreList = DataUtil.getPlotsGenre(d,genre_List)

	print("Random Forest Models with Word2Vec")		
	plotListEncoded = TFIDFEncoding.encodeData(plotList)
	randomForestTrainTest(plotListEncoded, genreList)

	print("Random Forest Models with TF-IDF")	
	Word2VecModel, tokenizedList = Word2VecEncoding.trainWord2VecModel(plotList)
	plotListWord2Vec = Word2VecEncoding.Word2VecEncode(tokenizedList,Word2VecModel)
	randomForestTrainTest(plotListWord2Vec,genreList)

	tokenizer = DistillBertUtil.getTokenizer()
	trainData, testData = DistillBertUtil.getDistillBertData(plotList, genreList, tokenizer)
	bertModel = DistillBertUtil.getDistillBert(3)
	print("Start Training")
	DistillBertUtil.trainModel(bertModel, trainData, testData, tokenizer)