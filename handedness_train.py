"""
    Maya Krolik
    Advanced Honors CS
    November 2022
    Data: https://www.kaggle.com/datasets/corrphilip/numeral-gestures?select=stroke.csv
"""

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import pandas as pd
import math

# ------------------------------------------------------------------------------
def main():
    
    # read processed data (see process_data.py)
    data = (shuffle(pd.read_csv('archive/data.csv'))).reset_index()

    # make quantity of right and left datapoints equal
    data_left = data[data["Handedness"] < 0.5]
    data_right = data[data["Handedness"] > 0.5]

    row_left,_ = data_left.shape
    index = math.trunc(row_left * 0.7)
    train_data_left = data_left[index:]
    train_data_right = data_right[index:]
    train_data = pd.concat([train_data_left, train_data_right], axis = 0)

    test_data_left = data_left[:index]
    test_data_right = data_right[:index]
    test_data = pd.concat([test_data_left, test_data_right], axis = 0)

    print(test_data)

    # split test data into dependent and independent variables
    testY = test_data.pop("Handedness")
    testX = test_data

    # split train data into dependent and independent variables
    trainY = train_data.pop("Handedness")
    trainX = train_data

    print(trainY)

    print(testY.shape)
    print(trainY.shape)

    # further split data into test and train data, making sure to shuffle for randomness
    # already did the hard work of seperateing test and training data, so test_size = 1
    # trainX, testX, trainY, testY = train_test_split(
    #     dataX, dataY, test_size = 1, shuffle = True
    #     )

    # create instance of LogisticRegression model and train
    # since the data is skewed (89% Right handed 11% Left handed), left (0) should be weighed around 8.09 times as much as right (1)
    # classifier = LogisticRegression(max_iter = 10000, class_weight= {0:8.09, 1:1})
    classifier = LogisticRegression(max_iter = 10000)
    classifier.fit(trainX, trainY)

    # create array of predictions using trained model
    preds = classifier.predict(testX)

    # assess accuracy of model
    score = classifier.score(testX, testY)
    print("certainty: " + str(score))

    # create and graph confusion matrix
    matrix = confusion_matrix(testY, preds)
    display = ConfusionMatrixDisplay(confusion_matrix = matrix, display_labels=["left", "right"])
    display.plot()
    plt.show()

main()