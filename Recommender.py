from surprise import SVD, BaselineOnly, NMF, SlopeOne
from surprise import Dataset
from surprise import Reader
from surprise.model_selection import cross_validate
from surprise import accuracy
from AdvancedRecommendations import RecommendationsOutput, calcAdvanced
import pandas as pd
import numpy as np

def main():
    #Reading the data and processing it to fetch the required attributes
    dataframe = pd.read_json('Software_5.json',lines= True)
    simplifieddataframe = dataframe[["overall", "reviewerID", "asin"]]
    print("Data processed")

    print("Total Number of ratings :",simplifieddataframe.shape[0])
    print("Total Number of Users   :", len(np.unique(simplifieddataframe.reviewerID)))
    print("Total Number of products  :", len(np.unique(simplifieddataframe.asin)))
    
    #Grouping data by reviewers
    reviewsperuser = simplifieddataframe.groupby(by='reviewerID')
    #Splitting available data to create training and testing sets
    trainingdata = reviewsperuser.sample(frac=0.8)
    testingdata = simplifieddataframe.drop(trainingdata.index)
    reader = Reader(rating_scale=(1,5))

    #Split the data into testing and training sets
    trainingset = Dataset.load_from_df(trainingdata[['reviewerID','asin','overall']], reader).build_full_trainset()
    testingset = Dataset.load_from_df(testingdata[['reviewerID','asin','overall']], reader).build_full_trainset().build_testset()
    
    #Algorithms
    SVDalgorithm = SVD(verbose=True, n_epochs=10)
    Baselinealgorithm = BaselineOnly({'method': 'sgd', 'n_epochs': 10, 'reg_u': 100, 'reg_i': 50},verbose=True)
    SlopeOnealgorithm = SlopeOne()
    NMFalgorithm = NMF()
    
    SVDalgorithm.fit(trainingset)
    Baselinealgorithm.fit(trainingset)
    SlopeOnealgorithm.fit(trainingset)
    NMFalgorithm.fit(trainingset)

    SVDpredictions = SVDalgorithm.test(testingset)
    
    Baselinepredictions = Baselinealgorithm.test(testingset)
    NMFpredictions = NMFalgorithm.test(testingset)
    SlopeOnepredictions = SlopeOnealgorithm.test(testingset)
    
    print("\nSVD metrics")
    print(accuracy.mae(SVDpredictions, verbose=True))
    print(accuracy.rmse(SVDpredictions, verbose=True))
    calcAdvanced(SVDpredictions)
    RecommendationsOutput(SVDpredictions, "SVDoutput")
    print("\n")
    print("Baseline metrics")
    print(accuracy.mae(Baselinepredictions, verbose=True))
    print(accuracy.rmse(Baselinepredictions, verbose=True))
    calcAdvanced(Baselinepredictions)
    RecommendationsOutput(Baselinepredictions, "Baselineoutput")
    print("\n")
    print("NMF metrics")
    print(accuracy.mae(NMFpredictions, verbose=True))
    print(accuracy.rmse(NMFpredictions, verbose=True))
    calcAdvanced(NMFpredictions)
    RecommendationsOutput(NMFpredictions, "NMFoutput")
    print("\n")
    print("SlopeOne metrics")
    print(accuracy.mae(SlopeOnepredictions, verbose=True))
    print(accuracy.rmse(SlopeOnepredictions, verbose=True))
    calcAdvanced(SlopeOnepredictions)
    RecommendationsOutput(SlopeOnepredictions, "SlopeOneoutput")

if __name__ == "__main__":
    main()