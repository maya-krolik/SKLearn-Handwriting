"""
    Maya Krolik
    Advanced Honors CS
    November 2022
    Data: https://www.kaggle.com/datasets/corrphilip/numeral-gestures?select=stroke.csv
"""

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import pandas as pd

# ------------------------------------------------------------------------------
def main():
    
    # read processed data (see process_data.py)
    data = pd.read_csv('archive/data.csv')

    # split data into dependent and independent variables
    dataY = data.pop("Gender")
    dataX = data

    # further split data into test and train data, making sure to shuffle for randomness
    trainX, testX, trainY, testY = train_test_split(
        dataX, dataY, test_size = 0.3, shuffle = True
        )

    # create instance of LogisticRegression model and train 
    classifier = LogisticRegression(max_iter = 10000)
    classifier.fit(trainX, trainY)

    # create array of predictions using trained model
    preds = classifier.predict(testX)

    # assess accuracy of model
    score = classifier.score(testX, testY)
    print("certainty: " + str(score))

    # create and graph confusion matrix
    matrix = confusion_matrix(testY, preds)
    display = ConfusionMatrixDisplay(confusion_matrix = matrix, display_labels=["female", "male"])
    display.plot()
    plt.show()

main()