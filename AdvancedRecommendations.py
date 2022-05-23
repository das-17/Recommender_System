#Extra Credit 
from collections import defaultdict
import pandas as pd

def calcAdvanced(predictions):
    # Mapping item recommendation by user
    groupedpredictions = defaultdict(list)
    for uid, iid, r_ui, est, details in predictions:
        groupedpredictions[uid].append((est, r_ui))

    precisions = dict()
    recalls = dict()
    
    for uid, uratings in groupedpredictions.items():
       TotalRelevant = 0
       RecommendedK = 0
       RelevantRecommendedK = 0
       #Sort in descending order
       uratings.sort(key=lambda x: x[0], reverse=True)

        # Number of True Positives and False Negatives
       for est, r_ui in uratings:
            if (r_ui > 3):
                TotalRelevant+=1
        
       for est, r_ui in uratings[:10]:
        # Number of TruePositive and FalsePositive
            if (est >= 3):
                RecommendedK+=1
        # Number of True Positives
            if (est >= 3) and (r_ui >= 3):
                RelevantRecommendedK+=1
       
       if RecommendedK != 0:
            precisions[uid] = RelevantRecommendedK / RecommendedK
       else:
            precisions[uid] = 0
       if TotalRelevant != 0:
           recalls[uid] = RelevantRecommendedK / TotalRelevant
       else:
           recalls[uid] = 0
    PrecisionSum = 0
    RecallSum = 0
    conversionR = 0
    for precision in precisions.values():
       PrecisionSum+=precision
    for recall in recalls.values():
        RecallSum+=recall
    totalPrecision = PrecisionSum/len(precisions)
    totalRecall = RecallSum/len(recalls)
    print("Precision:",totalPrecision)
    print("Recall:",totalRecall)
    print("F-Measure:",(2*totalRecall*totalPrecision)/(totalPrecision+totalRecall))
    for precision in precisions.values():
        if precision>0:
            conversionR+=1
    print("Conversion Rate:",conversionR/len(precisions.values()))
 
def RecommendationsOutput(predictions, outputFile):

    # Mapping item recommendation by user
    topRecommendations = defaultdict(list)
    for uid, iid, r_ui, est, details in predictions:
        topRecommendations[uid].append((iid, est))

   #Top available 10 item recommendations per user
    for uid, uratings in topRecommendations.items():
        uratings.sort(key=lambda x: x[1], reverse=True)
        topRecommendations[uid] = uratings[:10]
    File = open(outputFile, "w")
    for uid, user_ratings in topRecommendations.items():
        for i, (iid, est) in enumerate(user_ratings):
            File.write(uid+" "+iid+" "+str(i+1)+"\n")