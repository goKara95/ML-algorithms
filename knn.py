import numpy as np
import pandas as pd
import math
import random

#for mapping of weighted knn algorithm
label_dict = {
    "ESTJ": 0,
    "ENTJ": 1,
    "ESFJ": 2,
    "ENFJ": 3,
    "ISTJ": 4,
    "ISFJ": 5,
    "INTJ": 6,
    "INFJ": 7,
    "ESTP": 8,
    "ESFP": 9,
    "ENTP": 10,
    "ENFP": 11,
    "ISTP": 12,
    "ISFP": 13,
    "INTP": 14,
    "INFP": 15,
}

label_dict_reverse = {
    0: "ESTJ",
    1: "ENTJ",
    2: "ESFJ",
    3: "ENFJ",
    4: "ISTJ",
    5: "ISFJ",
    6: "INTJ",
    7: "INFJ",
    8: "ESTP",
    9: "ESFP",
    10: "ENTP",
    11: "ENFP",
    12: "ISTP",
    13: "ISFP",
    14: "INTP",
    15: "INFP",
}

def shuffle(array):
    #Since using randomly generated numbers are biased,
    #meaning the number of random outcomes of the algorithm don't have equal probabilities
    # We used fisher-yates shuffle algorithm: https://en.wikipedia.org/wiki/Fisher%E2%80%93Yates_shuffle.
    for i in range(len(array)-1, 0,-1):
        j = random.randrange(i)
        foo = np.copy(array[j])
        array[j] = np.copy(array[i])
        array[i] = foo
    return array


# calculates the euclidian distance between two points and returns the distance
def euclidean_distance(arr1, arr2):
    # (arr1-arr2) numpy function to subtract two arrays, it gives an array.
    # Then use another numpy function to take square of every index of the result array
    squareList = np.square(arr1 - arr2)
    # print(squareList)

    squareSum = 0
    for i in squareList:  # loop through array and add every index to variable
        squareSum += i

    # print(squareSum)
    return math.sqrt(squareSum)  # take square root


# min-max feature normalization
def minmax_normalization(data):
    dataMinMax = data.copy()  # copy to result array since we need a result array with the same size

    # we start the range with 1 since first column is index column
    # we end the range with (len of columns) - 1 since last column is label
    for i in range(1, len(data[0, :]) - 1):
        # i is corresponding to column numbers
        maxValue = np.max(data[:, i])  # get the max value for each column
        minValue = np.min(data[:, i])  # get the min value for each column

        for j in range(len(data[:, i])):
            # j is corresponding to row numbers
            dataMinMax[j][i] = (data[j][i] - minValue) / (maxValue - minValue)  # min max calculation

    return dataMinMax


# custom made metrics (precision, recall and accuracy)

# result is an array [index, predicted class, true class] generated from knn and weighted knn functions
# label_num is the count of class types
# this function returns an array [average precision, average recall, average accuracy]
def metrics(result, label_num):
    result = np.array(result)  # convert to numpy array to make slicing

    confusion_matrix = np.zeros((label_num, label_num))  # create empty array for confusion matrix

    y_true = result[:, 2]  # get the true classes of data as array
    y_pred = result[:, 1]  # get the predicted classes from knn functions as array

    # add the classes to confusion matrix
    # rows are actual classes, columns are predicted classes
    for i in range(len(y_true)):
        confusion_matrix[label_dict[y_true[i]]][label_dict[y_pred[i]]] += 1

    # create an 2d array with 16 rows and 4 columns
    # there are 16 row since there is 16 classes
    # column 1: true positive, column 2: false positive, column 3: true negative, column 4: false negative
    metric_classes = np.zeros((label_num, 4))

    # find true positive, false positive, true negative, false negative for every class type
    for i in range(label_num):
        # find the true positive
        tp = confusion_matrix[i][i]  # true positive is diagonal value
        metric_classes[i][0] = tp  # store the true positive in column 0

        # find the false positive
        # false positive for a class is the sum of values in the corresonding column exluding the true positive
        fpArr = np.sum(confusion_matrix,
                       axis=0)  # get the sum of every column as array (every index is the sum corresponding column)
        fp = fpArr[i] - tp  # get the sum of the desired column by accesing the array with using the corresponding index
        metric_classes[i][1] = fp  # store the true positive in column 1

        # we don't use true negative in this assignment
        # tn= np.sum(confusion_matrix) - ((np.sum(confusion_matrix[i]) + np.sum(confusion_matrix[:, i]) - confusion_matrix[i][i]))
        # metric_classes[i][2] = tn

        # find the false negative
        # false negative for a class is the sum of values in the corresonding row exluding the true positive
        fnArr = np.sum(confusion_matrix,
                       axis=1)  # get the sum of every row as array (every index is the sum corresponding row)
        fn = fnArr[i] - tp  # get the sum of the desired row by accesing the array with using the corresponding index
        metric_classes[i][3] = fn  # store the true positive in column 2

    # there are 16 row since there is 16 classes
    # column 1: precision, column 2: recall
    metrics = np.zeros((label_num, 2))
    for i in range(label_num):
        metrics[i][0] = metric_classes[i][0] / (
                    metric_classes[i][0] + metric_classes[i][1])  # precision = tp / (tp + fp)
        metrics[i][1] = metric_classes[i][0] / (metric_classes[i][0] + metric_classes[i][3])  # recall = tp / (to + fn)

    # calculate the average of every rows to get the average of every class type and round it to 2
    precision_avg = round((np.sum(metrics[:, 0]) / label_num), 5)
    recall_avg = round((np.sum(metrics[:, 1]) / label_num), 5)

    # find accuracy
    # accuracy = total true positive / total predicted class
    total = np.sum(confusion_matrix)  # sum of every element of confusion matrix
    # total true positive is sum of all diagonals of confusion matrix
    totalTruePositive = np.trace(confusion_matrix)  # np.trace() function gives the sum of all diagonals
    accuracy = round(totalTruePositive / total, 5)

    # return this values as array
    return [precision_avg, recall_avg, accuracy]


def split(arr, fold):
    # Takes an array and fold number as argument.
    # splits to array in to even chunks(if len is power of 5, otherwise last set will have more elements)
    sets = []
    block_size = int(len(arr) / fold)
    for i in range(fold):
        if i == 0:
            sets.append(arr[:block_size])
        elif i == fold - 1:
            sets.append(arr[block_size * i:])
        else:
            sets.append(arr[block_size * i: block_size * (i + 1)])

    return sets


# Takes 5 set as argument, since we will split dataset to 5 chunks
# set1 is the test set, while traversing set1 we traverse every other set and calculate eucledean distance between each element
# the array dist holds the distance between two elements and the class of the instance that belongs to train data, that way
def knn(set1, set2, set3, set4, set5):
    knn = []
    w_knn = []
    dist = []

    def most_common(lst):
        return max(set(lst), key=lst.count)

    classes = []
    true = 0

    # these arrays contain prediction arrays for each instance in set1, k1Total is predicted with 1 neighbor and so on...
    # k1[i][0] = data id, k1[i][1] = real class, k1[i][2] = predicted class, k1[i][3]=whether the prediction is true or not
    k1Total = []
    k3Total = []
    k5Total = []
    k7Total = []
    k9Total = []

    for i in range(len(set1)):
        # euclidean distance and class of train instance is added to dist array, and this array is appended to knn array
        # after executing the loops below knn[] will hold information of the distance between current element and every other
        # instance in train sets
        for j in range(len(set2)):
            dist.append(euclidean_distance(set1[i][1:61], set2[j][1:61]))
            dist.append(set2[j][61])
            knn.append(dist)
            dist = []

        for k in range(len(set3)):
            dist.append(euclidean_distance(set1[i][1:61], set3[k][1:61]))
            dist.append(set3[k][61])
            knn.append(dist)
            dist = []

        for z in range(len(set4)):
            dist.append(euclidean_distance(set1[i][1:61], set4[z][1:61]))
            dist.append(set4[z][61])
            knn.append(dist)
            dist = []

        for l in range(len(set5)):
            dist.append(euclidean_distance(set1[i][1:61], set5[l][1:61]))
            dist.append(set5[l][61])
            knn.append(dist)
            dist = []
        # knn is sorted in ascending order. Meaning closest train instance to our test instance will be in the first position
        # since knn is an array of arrays note that knn[0][0] is the distance between test instance and closest train instance
        # while knn[0][1] is the class of the train instance
        knn = sorted(knn)

        # we traverse knn array and add class names of closest neighbours to classes list.
        # That way after kth iteration we will get the most occuring element in the classes using most common function above.

        # for k=1
        classes.append(knn[0][1])
        k1Arr = [set1[i][0], knn[0][1], set1[i][61]]
        k1Total.append(k1Arr)

        # for k=3
        for r in range(1, 3):
            classes.append(knn[r][1])
        k3Arr = [set1[i][0], most_common(classes), set1[i][61]]
        k3Total.append(k3Arr)

        # for k=5
        for r in range(3, 5):
            classes.append(knn[r][1])
        k5Arr = [set1[i][0], most_common(classes), set1[i][61]]
        k5Total.append(k5Arr)

        # for k=7
        for r in range(5, 7):
            classes.append(knn[r][1])
        k7Arr = [set1[i][0], most_common(classes), set1[i][61]]
        k7Total.append(k7Arr)

        # for k=9
        for r in range(7, 9):
            classes.append(knn[r][1])
        k9Arr = [set1[i][0], most_common(classes), set1[i][61]]
        k9Total.append(k9Arr)

        classes = []
        knn = []
    return [k1Total, k3Total, k5Total, k7Total, k9Total]


# take 5 set that splited and label_num is the count of class types
def weighted_knn(set1, set2, set3, set4, set5, label_num):
    # weighted knn
    # result arrays
    k1Total = []
    k3Total = []
    k5Total = []
    k7Total = []
    k9Total = []

    knn = []  # to store classes with their distances
    classes = []  # to store the nearest knn points (number of points will be stored depending on the number of k)
    dist = []

    # create w_classes array to find the maximum total weight (indexes corresponds to classes)
    w_classes = []

    for i in range(label_num):
        w_classes.append(0)

    # until "knn = sorted(knn)" line it is same with the normal (unweighted) knn
    for i in range(len(set1)):
        for j in range(len(set2)):
            dist.append(euclidean_distance(set1[i][1:61], set2[j][1:61]))
            dist.append(set2[j][61])
            knn.append(dist)
            dist = []

        for k in range(len(set3)):
            dist.append(euclidean_distance(set1[i][1:61], set3[k][1:61]))
            dist.append(set3[k][61])
            knn.append(dist)
            dist = []

        for z in range(len(set4)):
            dist.append(euclidean_distance(set1[i][1:61], set4[z][1:61]))
            dist.append(set4[z][61])
            knn.append(dist)
            dist = []

        for l in range(len(set5)):
            dist.append(euclidean_distance(set1[i][1:61], set5[l][1:61]))
            dist.append(set5[l][61])
            knn.append(dist)
            dist = []
        knn = sorted(knn)

        # for k=1  (same with unweighted)
        classes.append(knn[0][1])  # add nearst point (class)

        # first is index, second is predicted class, third is label (actual class)
        if set1[i][61] == knn[0][1]:
            k1Arr = [set1[i][0], knn[0][1], set1[i][61]]
        else:
            k1Arr = [set1[i][0], knn[0][1], set1[i][61]]
        k1Total.append(k1Arr)

        # for k=3
        k3_weight0_flag = False  # flag to check if we found a point with 0 distance
        # this helps stop checking the maximum weight later

        # add second and third points to array that already holds the most nearst point
        for r in range(1, 3):
            classes.append(knn[r][1])

        for m in range(len(classes)):
            if (knn[m][0] != 0):  # if distance isn't 0, calculate the weight of the point
                weight = 1.0 / knn[m][0]  # get the inverse of distance to acquire weight

                # main point of it: w_classes's index is corresponds to number of class and it's value stores the weight of that class
                # For example: let's say w_classes[2] = 14.6 this means class number 2 ("ESFJ") has a weight of 14.6
                w_classes[
                    label_dict[knn[m][1]]] += weight  # increment the weight in the correspending index of the array
            else:
                # if distance is 0, no need to calculate the weight. predictedClass is the class with 0 distance
                predictedClass = knn[m][1]
                k3_weight0_flag = True
                break  # since we don't need to look other points, stop the loop

        # if we don't find a point with 0 distance, find the max weight
        # else don't go in to this block and let predictedClass stay as it is
        if k3_weight0_flag == False:
            maxWeight = max(w_classes)  # get the max value of between the values of array
            # get the index that holds the max value since that is our predicted class
            predictedClass = label_dict_reverse[w_classes.index(maxWeight)]

            # add the results to array
        if predictedClass == set1[i][61]:
            k3Arr = [set1[i][0], predictedClass, set1[i][61]]
        else:
            k3Arr = [set1[i][0], predictedClass, set1[i][61]]
        k3Total.append(k3Arr)

        # k5, k7 and k9 is same with k3 only change is the number of nearst neighbors we evalute

        # for k=5
        k5_weight0_flag = False
        for r in range(3, 5):
            classes.append(knn[r][1])

        for m in range(len(classes)):
            if (knn[m][0] != 0):
                weight = 1.0 / knn[m][0]
                w_classes[label_dict[knn[m][1]]] += weight
            else:
                predictClass = knn[m][1]
                k5_weight0_flag = True
                break

        if k5_weight0_flag == False:
            maxWeight = max(w_classes)
            predictedClass = label_dict_reverse[w_classes.index(maxWeight)]

        if predictedClass == set1[i][61]:
            k5Arr = [set1[i][0], predictedClass, set1[i][61]]
        else:
            k5Arr = [set1[i][0], predictedClass, set1[i][61]]
        k5Total.append(k5Arr)

        # for k=7
        k7_weight0_flag = False
        for r in range(5, 7):
            classes.append(knn[r][1])

        for m in range(len(classes)):
            if (knn[m][0] != 0):
                weight = 1.0 / knn[m][0]
                w_classes[label_dict[knn[m][1]]] += weight
            else:
                predictedClass = knn[m][1]
                k7_weight0_flag = True
                break

        if k7_weight0_flag == False:
            maxWeight = max(w_classes)
            predictedClass = label_dict_reverse[w_classes.index(maxWeight)]

        if predictedClass == set1[i][61]:
            k7Arr = [set1[i][0], predictedClass, set1[i][61]]
        else:
            k7Arr = [set1[i][0], predictedClass, set1[i][61]]
        k7Total.append(k7Arr)

        # for k=9
        k9_weight0_flag = False
        for r in range(7, 9):
            classes.append(knn[r][1])

        for m in range(len(classes)):
            if (knn[m][0] != 0):
                weight = 1.0 / knn[m][0]
                w_classes[label_dict[knn[m][1]]] += weight
            else:
                predictedClass = knn[m][1]
                k9_weight0_flag = True
                break

        if k9_weight0_flag == False:
            maxWeight = max(w_classes)
            predictedClass = label_dict_reverse[w_classes.index(maxWeight)]

        if predictedClass == set1[i][61]:
            k9Arr = [set1[i][0], predictedClass, set1[i][61], 1]
        else:
            k9Arr = [set1[i][0], predictedClass, set1[i][61], 0]
        k9Total.append(k9Arr)

        classes = []
        knn = []
        w_classes = []

        for i in range(16):
            w_classes.append(0)

    # returns result for k1, k3, k5, k7 and k9 as an array
    return [k1Total, k3Total, k5Total, k7Total, k9Total]

personDataFrame = pd.read_csv("subset_16P.csv", encoding='unicode_escape')
print(personDataFrame)

personArr = personDataFrame.to_numpy()
print(personArr)

shuffle(personArr)
print(personArr)

#Proof of normalization process
normalizedPersonArr = minmax_normalization(personArr)
print(normalizedPersonArr)

# split the data
sets = split(personArr, 5)
set1 = sets[0]
set2 = sets[1]
set3 = sets[2]
set4 = sets[3]
set5 = sets[4]

normalized_sets = split(normalizedPersonArr, 5)
normalized_set1 = normalized_sets[0]
normalized_set2 = normalized_sets[1]
normalized_set3 = normalized_sets[2]
normalized_set4 = normalized_sets[3]
normalized_set5 = normalized_sets[4]

#set1 is test data
# fold1 raw data (using set1 as test)
knn_results1 = knn(set1,set2,set3,set4,set5)
weighted_knn_results1 = weighted_knn(set1,set2,set3,set4,set5, 16)

# fold1 normalized data (using set1 as test)
normalized_knn_results1 = knn(normalized_set1, normalized_set2, normalized_set3, normalized_set4, normalized_set5)
normalized_weighted_knn_results1 = weighted_knn(normalized_set1, normalized_set2, normalized_set3,
                                               normalized_set4, normalized_set5, 16)

#set2 is test data
knn_results2 = knn(set2,set1,set3,set4,set5)
weighted_knn_results2 = weighted_knn(set2,set1,set3,set4,set5, 16)

normalized_knn_results2 = knn(normalized_set2, normalized_set1, normalized_set3, normalized_set4, normalized_set5)
normalized_weighted_knn_results2 = weighted_knn(normalized_set2, normalized_set1, normalized_set3,
                                               normalized_set4, normalized_set5, 16)

# set3 is test data
knn_results3 = knn(set3,set1,set2,set4,set5)
weighted_knn_results3 = weighted_knn(set3,set1,set2,set4,set5, 16)

normalized_knn_results3 = knn(normalized_set3, normalized_set1, normalized_set2, normalized_set4, normalized_set5)
normalized_weighted_knn_results3 = weighted_knn(normalized_set3, normalized_set1, normalized_set2,
                                               normalized_set4, normalized_set5, 16)

# set4 is test data
knn_results4 = knn(set4,set1,set2,set3,set5)
weighted_knn_results4 = weighted_knn(set4,set1,set2,set3,set5, 16)

normalized_knn_results4 = knn(normalized_set4, normalized_set1, normalized_set2, normalized_set3, normalized_set5)
normalized_weighted_knn_results4 = weighted_knn(normalized_set4, normalized_set1, normalized_set2,
                                               normalized_set3, normalized_set5, 16)

# set5 is test data
knn_results5 = knn(set5,set1,set2,set3,set4)
weighted_knn_results5 = weighted_knn(set5,set1,set2,set3,set4, 16)

normalized_knn_results5 = knn(normalized_set5, normalized_set1, normalized_set2, normalized_set3, normalized_set4)
normalized_weighted_knn_results5 = weighted_knn(normalized_set5, normalized_set1, normalized_set2,
                                               normalized_set3, normalized_set4, 16)

pd.set_option('display.precision', 3)
# precision
precision = np.zeros((5, 20))
# row 0 is fold 1, row 1 is fold 2, row 2 is fold 3, row 3 is fold 4, row 5 is fold 5
# there is 20 column: column 0-5 is raw data knn, column 5-10 is raw data weighted knn,
# column 10-15 is normalized data knn, column 15-20 is normalized data weighted knn

# fold 1

# raw data knn
for i in range(5):  # enter the raw data knn to the first 5 column
    # execute the metrics function with the every index of knn_results array (knn_Result[0] is k=1, knn_Result[1] is k=3 and so on)
    # then take the first index of the result of metrics function since we want precision (metrics(arg)[0] is precision)
    asd = metrics(knn_results1[i], 16)[0]
    # store it to precision array row 0 and correspending column
    precision[0][i] = asd

# raw data weighted knn
for i in range(5, 10):  # enter the raw data weighted knn to the 5-10 column
    # we execute the metrics function with i-5 since we need to acces knn_results array
    # For example i = 5 but we need to acces 0th index of weighted_knn_results1 to get k= 1
    # then take the first index of the result of metrics function like we did in raw data knn
    asd = metrics(weighted_knn_results1[i - 5], 16)[0]
    # store the average precison to array's row 0 and correspending column
    precision[0][i] = asd

# normalized data knn
for i in range(10, 15):  # enter the normalized data knn to the 10-15 column
    # it is same with raw data weighted knn, only change is we substract 10 from i since range starts with 10
    asd = metrics(normalized_knn_results1[i - 10], 16)[0]
    precision[0][i] = asd

# normalized data weighted knn
for i in range(15, 20):  # enter the normalized data weighted knn to the 15-20 column
    # it is same with raw data weighted knn, only change is we substract 15 from i since range starts with 15
    asd = metrics(normalized_weighted_knn_results1[i - 15], 16)[0]
    precision[0][i] = asd

# fold 2
# it is same with fold 1, only change is we store the average precision to array's row 1 and corresponding column

# raw data knn
for i in range(5):
    asd = metrics(knn_results2[i], 16)[0]
    precision[1][i] = asd

# raw data weighted knn
for i in range(5, 10):
    asd = metrics(weighted_knn_results2[i - 5], 16)[0]
    precision[1][i] = asd

# normalized data knn
for i in range(10, 15):
    asd = metrics(normalized_knn_results2[i - 10], 16)[0]
    precision[1][i] = asd

# normalized data weighted knn
for i in range(15, 20):
    asd = metrics(normalized_weighted_knn_results2[i - 15], 16)[0]
    precision[1][i] = asd

# fold 3

# raw data knn
for i in range(5):
    asd = metrics(knn_results3[i], 16)[0]
    precision[2][i] = asd

# raw data weighted knn
for i in range(5, 10):
    asd = metrics(weighted_knn_results3[i - 5], 16)[0]
    precision[2][i] = asd

# normalized data knn
for i in range(10, 15):
    asd = metrics(normalized_knn_results3[i - 10], 16)[0]
    precision[2][i] = asd

# normalized data weighted knn
for i in range(15, 20):
    asd = metrics(normalized_weighted_knn_results3[i - 15], 16)[0]
    precision[2][i] = asd

# fold 4

# raw data knn
for i in range(5):
    asd = metrics(knn_results4[i], 16)[0]
    precision[3][i] = asd

# raw data weighted knn
for i in range(5, 10):
    asd = metrics(weighted_knn_results4[i - 5], 16)[0]
    precision[3][i] = asd

# normalized data knn
for i in range(10, 15):
    asd = metrics(normalized_knn_results4[i - 10], 16)[0]
    precision[3][i] = asd

# normalized data weighted knn
for i in range(15, 20):
    asd = metrics(normalized_weighted_knn_results4[i - 15], 16)[0]
    precision[3][i] = asd

# fold 5

# raw data knn
for i in range(5):
    asd = metrics(knn_results5[i], 16)[0]
    precision[4][i] = asd

# raw data weighted knn
for i in range(5, 10):
    asd = metrics(weighted_knn_results5[i - 5], 16)[0]
    precision[4][i] = asd

# normalized data knn
for i in range(10, 15):
    asd = metrics(normalized_knn_results5[i - 10], 16)[0]
    precision[4][i] = asd

# normalized data weighted knn
for i in range(15, 20):
    asd = metrics(normalized_weighted_knn_results5[i - 15], 16)[0]
    precision[4][i] = asd

precision = np.vstack([precision, np.mean(precision, axis=0)])

precision_df = pd.DataFrame(precision, columns=['k=1, knn, raw data', 'k=3, knn, raw data', 'k=5, knn, raw data',
                                                "k=7, knn, raw data",
                                                'k=9, knn, raw data', 'k=1, w-knn, raw data', 'k=3, w-knn, raw data',
                                                "k=5, w-knn, raw data",
                                                'k=7, w-knn, raw data', 'k=9, w-knn, raw data',
                                                'k=1, knn, normalized data', 'k=3, knn, normalized',
                                                'k=5, knn, normalized', "k=7, knn, normalized",
                                                'k=9, knn, normalized', 'k=1, w-knn, normalized',
                                                'k=3, w-knn, normalized', "k=5, w-knn, normalized",
                                                'k=7, w-knn, normalized', 'k=9, w-knn, normalized'])
precision_df.index = ["Fold1", "Fold2", "Fold3", "Fold4", "Fold5", "Average"]
#precision_df = precision_df.style.set_caption("Precision Table for 5 Fold Cross-Validation")
print(precision_df)

# recall
recall = np.zeros((5, 20))

# it is same with precision

# fold 1

# raw data normal knn
for i in range(5):
    asd = metrics(knn_results1[i], 16)[1]
    recall[0][i] = asd

# raw data weighted knn
for i in range(5, 10):
    asd = metrics(weighted_knn_results1[i - 5], 16)[1]
    recall[0][i] = asd

# normalized data knn
for i in range(10, 15):
    asd = metrics(normalized_knn_results1[i - 10], 16)[1]
    recall[0][i] = asd

# normalized data weighted knn
for i in range(15, 20):
    asd = metrics(normalized_weighted_knn_results1[i - 15], 16)[1]
    recall[0][i] = asd

# fold 2

# raw data normal knn
for i in range(5):
    asd = metrics(knn_results2[i], 16)[1]
    recall[1][i] = asd

# raw data weighted knn
for i in range(5, 10):
    asd = metrics(weighted_knn_results2[i - 5], 16)[1]
    recall[1][i] = asd

# normalized data knn
for i in range(10, 15):
    asd = metrics(normalized_knn_results2[i - 10], 16)[1]
    recall[1][i] = asd

# normalized data weighted knn
for i in range(15, 20):
    asd = metrics(normalized_weighted_knn_results2[i - 15], 16)[1]
    recall[1][i] = asd

# fold 3

# raw data normal knn
for i in range(5):
    asd = metrics(knn_results3[i], 16)[1]
    recall[2][i] = asd

# raw data weighted knn
for i in range(5, 10):
    asd = metrics(weighted_knn_results3[i - 5], 16)[1]
    recall[2][i] = asd

# normalized data knn
for i in range(10, 15):
    asd = metrics(normalized_knn_results3[i - 10], 16)[1]
    recall[2][i] = asd

# normalized data weighted knn
for i in range(15, 20):
    asd = metrics(normalized_weighted_knn_results3[i - 15], 16)[1]
    recall[2][i] = asd

# fold 4

# raw data normal knn
for i in range(5):
    asd = metrics(knn_results4[i], 16)[1]
    recall[3][i] = asd

# raw data weighted knn
for i in range(5, 10):
    asd = metrics(weighted_knn_results4[i - 5], 16)[1]
    recall[3][i] = asd

# normalized data knn
for i in range(10, 15):
    asd = metrics(normalized_knn_results4[i - 10], 16)[1]
    recall[3][i] = asd

# normalized data weighted knn
for i in range(15, 20):
    asd = metrics(normalized_weighted_knn_results4[i - 15], 16)[1]
    recall[3][i] = asd

# fold 5

# raw data normal knn
for i in range(5):
    asd = metrics(knn_results5[i], 16)[1]
    recall[4][i] = asd

# raw data weighted knn
for i in range(5, 10):
    asd = metrics(weighted_knn_results5[i - 5], 16)[1]
    recall[4][i] = asd

# normalized data knn
for i in range(10, 15):
    asd = metrics(normalized_knn_results5[i - 10], 16)[1]
    recall[4][i] = asd

# normalized data weighted knn
for i in range(15, 20):
    asd = metrics(normalized_weighted_knn_results5[i - 15], 16)[1]
    recall[4][i] = asd

recall = np.vstack([recall, np.mean(recall, axis=0)])

recall_df = pd.DataFrame(recall, columns=['k=1, knn, raw data', 'k=3, knn, raw data', 'k=5, knn, raw data',
                                          "k=7, knn, raw data",
                                          'k=9, knn, raw data', 'k=1, w-knn, raw data', 'k=3, w-knn, raw data',
                                          "k=5, w-knn, raw data",
                                          'k=7, w-knn, raw data', 'k=9, w-knn, raw data',
                                          'k=1, knn, normalized data', 'k=3, knn, normalized', 'k=5, knn, normalized',
                                          "k=7, knn, normalized",
                                          'k=9, knn, normalized', 'k=1, w-knn, normalized', 'k=3, w-knn, normalized',
                                          "k=5, w-knn, normalized",
                                          'k=7, w-knn, normalized', 'k=9, w-knn, normalized'])
recall_df.index = ["Fold1", "Fold2", "Fold3", "Fold4", "Fold5", "Average"]
#recall_df = recall_df.style.set_caption("Recall Table for 5 Fold Cross-Validation")
print(recall_df)

# accuracy

# it is same with precision
accuracy = np.zeros((5, 20))

# fold 1

# raw data normal knn
for i in range(5):
    asd = metrics(knn_results1[i], 16)[2]
    accuracy[0][i] = asd

# raw data weighted knn
for i in range(5, 10):
    asd = metrics(weighted_knn_results1[i - 5], 16)[2]  # i-5 since this is weighted
    accuracy[0][i] = asd

# normalized data knn
for i in range(10, 15):
    asd = metrics(normalized_knn_results1[i - 10], 16)[2]
    accuracy[0][i] = asd

# normalized data weighted knn
for i in range(15, 20):
    asd = metrics(normalized_weighted_knn_results1[i - 15], 16)[2]
    accuracy[0][i] = asd

# fold 2

# raw data normal knn
for i in range(5):
    asd = metrics(knn_results2[i], 16)[2]
    accuracy[1][i] = asd

# raw data weighted knn
for i in range(5, 10):
    asd = metrics(weighted_knn_results2[i - 5], 16)[2]  # i-5 since this is weighted
    accuracy[1][i] = asd

# normalized data knn
for i in range(10, 15):
    asd = metrics(normalized_knn_results2[i - 10], 16)[2]
    accuracy[1][i] = asd

# normalized data weighted knn
for i in range(15, 20):
    asd = metrics(normalized_weighted_knn_results2[i - 15], 16)[2]
    accuracy[1][i] = asd

# fold 3

# raw data normal knn
for i in range(5):
    asd = metrics(knn_results3[i], 16)[2]
    accuracy[2][i] = asd

# raw data weighted knn
for i in range(5, 10):
    asd = metrics(weighted_knn_results3[i - 5], 16)[2]  # i-5 since this is weighted
    accuracy[2][i] = asd

# normalized data knn
for i in range(10, 15):
    asd = metrics(normalized_knn_results3[i - 10], 16)[2]
    accuracy[2][i] = asd

# normalized data weighted knn
for i in range(15, 20):
    asd = metrics(normalized_weighted_knn_results3[i - 15], 16)[2]
    accuracy[2][i] = asd

# fold 4

# raw data normal knn
for i in range(5):
    asd = metrics(knn_results4[i], 16)[2]
    accuracy[3][i] = asd

# raw data weighted knn
for i in range(5, 10):
    asd = metrics(weighted_knn_results4[i - 5], 16)[2]  # i-5 since this is weighted
    accuracy[3][i] = asd

# normalized data knn
for i in range(10, 15):
    asd = metrics(normalized_knn_results4[i - 10], 16)[2]
    accuracy[3][i] = asd

# normalized data weighted knn
for i in range(15, 20):
    asd = metrics(normalized_weighted_knn_results4[i - 15], 16)[2]
    accuracy[3][i] = asd

# fold 5

# raw data normal knn
for i in range(5):
    asd = metrics(knn_results5[i], 16)[2]
    accuracy[4][i] = asd

# raw data weighted knn
for i in range(5, 10):
    asd = metrics(weighted_knn_results5[i - 5], 16)[2]  # i-5 since this is weighted
    accuracy[4][i] = asd

# normalized data knn
for i in range(10, 15):
    asd = metrics(normalized_knn_results5[i - 10], 16)[2]
    accuracy[4][i] = asd

# normalized data weighted knn
for i in range(15, 20):
    asd = metrics(normalized_weighted_knn_results5[i - 15], 16)[2]
    accuracy[4][i] = asd

accuracy = np.vstack([accuracy, np.mean(accuracy, axis=0)])
accuracy_df = pd.DataFrame(accuracy, columns=['k=1, knn, raw data', 'k=3, knn, raw data', 'k=5, knn, raw data',
                                              "k=7, knn, raw data",
                                              'k=9, knn, raw data', 'k=1, w-knn, raw data', 'k=3, w-knn, raw data',
                                              "k=5, w-knn, raw data",
                                              'k=7, w-knn, raw data', 'k=9, w-knn, raw data',
                                              'k=1, knn, normalized data', 'k=3, knn, normalized',
                                              'k=5, knn, normalized', "k=7, knn, normalized",
                                              'k=9, knn, normalized', 'k=1, w-knn, normalized',
                                              'k=3, w-knn, normalized', "k=5, w-knn, normalized",
                                              'k=7, w-knn, normalized', 'k=9, w-knn, normalized'])
accuracy_df.index = ["Fold1", "Fold2", "Fold3", "Fold4", "Fold5", "Average"]
#accuracy_df = accuracy_df.style.set_caption("Accuracy Table for 5 Fold Cross-Validation")
print(accuracy_df)



df = pd.read_csv('energy_efficiency_data.csv')
dataset = np.array(df)
df.head(5)

#proof that shuffling is succesful
shuffle(dataset)
pd.DataFrame(dataset)


# min-max feature normalization
def energy_minmax_normalization(data):
    dataMinMax = np.copy(data)  # copy to result array since we need a result array with the same size

    for i in range(0, len(data[0, :]) - 2):
        maxValue = np.max(data[:, i])  # get the max value for each column
        minValue = np.min(data[:, i])  # get the min value for each column

        for j in range(len(data[:, i])):
            dataMinMax[j][i] = (data[j][i] - minValue) / (maxValue - minValue)  # min max calculation

    return dataMinMax


# Since this problem is about regression and not classification I will have to re-define knn and weighted knn functions
# knn = an array which holds 3 element arrays.
# 1st element distance between 2 instances, 2nd: train data's HL, 3rd: train data's CL
# set1 is used to test the model
def nearestNeighbour(set1, set2, set3, set4, set5):
    k1Total = []
    k3Total = []
    k5Total = []
    k7Total = []
    k9Total = []
    knn = []
    dist = []
    total_hl = 0
    total_cl = 0
    for i in range(len(set1)):  # outer loop
        for j in range(len(set2)):
            dist.append(euclidean_distance(set1[i][:8], set2[j][:8]))
            dist.append(set2[j][8])
            dist.append(set2[j][9])
            knn.append(dist)
            dist = []

        for k in range(len(set3)):
            dist.append(euclidean_distance(set1[i][:8], set3[k][:8]))
            dist.append(set3[k][8])
            dist.append(set3[k][9])
            knn.append(dist)
            dist = []

        for z in range(len(set4)):
            dist.append(euclidean_distance(set1[i][:8], set4[z][:8]))
            dist.append(set4[z][8])
            dist.append(set4[z][9])
            knn.append(dist)
            dist = []

        for l in range(len(set5)):
            dist.append(euclidean_distance(set1[i][:8], set5[l][:8]))
            dist.append(set5[l][8])
            dist.append(set5[l][9])
            knn.append(dist)
            dist = []
        knn = sorted(knn)

        # for k=1 k1Arr=predicted HL, predicted CL, real HL, real CL,
        k1Arr = [knn[0][1], knn[0][2], set1[i][8], set1[i][9]]
        k1Total.append(k1Arr)

        # for k=3,5,7,9
        # travel the knn array and find total hl/cl. Divide by k to make prediction
        for r in range(9):
            total_hl += knn[r][1]
            total_cl += knn[r][2]
            if r == 2:
                k3Arr = [total_hl / (3.0), total_cl / (3.0), set1[i][8], set1[i][9]]
                k3Total.append(k3Arr)

            if r == 4:
                k5Arr = [total_hl / (5.0), total_cl / (5.0), set1[i][8], set1[i][9]]
                k5Total.append(k5Arr)

            if r == 6:
                k7Arr = [total_hl / (7.0), total_cl / (7.0), set1[i][8], set1[i][9]]
                k7Total.append(k7Arr)

            if r == 8:
                k9Arr = [total_hl / (9.0), total_cl / (9.0), set1[i][8], set1[i][9]]
                k9Total.append(k9Arr)

        knn = []
        total_hl = 0
        total_cl = 0
    return [k1Total, k3Total, k5Total, k7Total, k9Total]


def weightedKnn(set1, set2, set3, set4, set5):
    k1Total = []
    k3Total = []
    k5Total = []
    k7Total = []
    k9Total = []
    knn = []
    dist = []
    total_hl = 0
    total_cl = 0
    total_w = 0
    for i in range(len(set1)):
        for j in range(len(set2)):
            dist.append(euclidean_distance(set1[i][:8], set2[j][:8]))
            dist.append(set2[j][8])
            dist.append(set2[j][9])
            knn.append(dist)
            dist = []

        for k in range(len(set3)):
            dist.append(euclidean_distance(set1[i][:8], set3[k][:8]))
            dist.append(set3[k][8])
            dist.append(set3[k][9])
            knn.append(dist)
            dist = []

        for z in range(len(set4)):
            dist.append(euclidean_distance(set1[i][:8], set4[z][:8]))
            dist.append(set4[z][8])
            dist.append(set4[z][9])
            knn.append(dist)
            dist = []

        for l in range(len(set5)):
            dist.append(euclidean_distance(set1[i][:8], set5[l][:8]))
            dist.append(set5[l][8])
            dist.append(set5[l][9])
            knn.append(dist)
            dist = []
        knn = sorted(knn)

        # for k=1 k1Arr=predicted HL, predicted CL, real HL, real CL,
        k1Arr = [knn[0][1], knn[0][2], set1[i][8], set1[i][9]]
        k1Total.append(k1Arr)

        # for k=3,5,7,9
        # travel the knn array and find total hl/cl. Divide by k to make prediction
        for r in range(9):
            total_hl += (1 / knn[r][0]) * knn[r][1]
            total_cl += (1 / knn[r][0]) * knn[r][2]
            total_w += 1 / knn[r][0]
            if r == 2:
                k3Arr = [total_hl / total_w, total_cl / total_w, set1[i][8], set1[i][9]]
                k3Total.append(k3Arr)

            if r == 4:
                k5Arr = [total_hl / total_w, total_cl / total_w, set1[i][8], set1[i][9]]
                k5Total.append(k5Arr)

            if r == 6:
                k7Arr = [total_hl / total_w, total_cl / total_w, set1[i][8], set1[i][9]]
                k7Total.append(k7Arr)

            if r == 8:
                k9Arr = [total_hl / total_w, total_cl / total_w, set1[i][8], set1[i][9]]
                k9Total.append(k9Arr)

        knn = []
        total_hl = 0
        total_cl = 0
        total_w = 0
    return [k1Total, k3Total, k5Total, k7Total, k9Total]


def MAE(array):
    arr = []  # index 0 = HL error, 1=CL error
    diff_hl = 0
    diff_cl = 0
    for i in range(len(array)):
        diff_hl += abs(array[i][0] - array[i][2])
        diff_cl += abs(array[i][1] - array[i][3])
    arr.append(diff_hl / len(array))
    arr.append(diff_cl / len(array))
    return arr

#splitting the dataset to 5 chunks since we will do 5-fold cv
#split function is from part1

n_dataset = energy_minmax_normalization(dataset)

#chunks for raw-data
splitted = split(dataset,5)
set1 = splitted[0]
set2 = splitted[1]
set3 = splitted[2]
set4 = splitted[3]
set5 = splitted[4]

#chunks for normalized data
n_splitted = split(n_dataset, 5)
n_set1 = n_splitted[0]
n_set2 = n_splitted[1]
n_set3 = n_splitted[2]
n_set4 = n_splitted[3]
n_set5 = n_splitted[4]

HL_error = np.zeros((5, 20))  # table with folds in rows and variations in columns
CL_error = np.zeros((5, 20))

# set1 is test data
fold1_knn = nearestNeighbour(set1, set2, set3, set4, set5)
fold1_wknn = weightedKnn(set1, set2, set3, set4, set5)
fold1_knn_normal = nearestNeighbour(n_set1, n_set2, n_set3, n_set4, n_set5)
fold1_wknn_normal = weightedKnn(n_set1, n_set2, n_set3, n_set4, n_set5)

# we iterate through 4 different for loops and add MAE to columns for 4 different variations.(since each variation has 5 k-value we have total of 20 cols)
for i in range(5):
    error = MAE(fold1_knn[i])
    HL_error[0][i] = error[0]
    CL_error[0][i] = error[1]

for i in range(5):
    error = MAE(fold1_wknn[i])
    HL_error[0][i + 5] = error[0]
    CL_error[0][i + 5] = error[1]

for i in range(5):
    error = MAE(fold1_knn_normal[i])
    HL_error[0][i + 10] = error[0]
    CL_error[0][i + 10] = error[1]

for i in range(5):
    error = MAE(fold1_wknn_normal[i])
    HL_error[0][i + 15] = error[0]
    CL_error[0][i + 15] = error[1]

# set2 is test data
fold2_knn = nearestNeighbour(set2, set1, set3, set4, set5)
fold2_wknn = weightedKnn(set2, set1, set3, set4, set5)
fold2_knn_normal = nearestNeighbour(n_set2, n_set1, n_set3, n_set4, n_set5)
fold2_wknn_normal = weightedKnn(n_set2, n_set1, n_set3, n_set4, n_set5)

# we iterate through 4 different for loops and add MAE to columns for 4 different variations.(since each variation has 5 k-value we have total of 20 cols)
for i in range(5):
    error = MAE(fold2_knn[i])
    HL_error[1][i] = error[0]
    CL_error[1][i] = error[1]

for i in range(5):
    error = MAE(fold2_wknn[i])
    HL_error[1][i + 5] = error[0]
    CL_error[1][i + 5] = error[1]

for i in range(5):
    error = MAE(fold2_knn_normal[i])
    HL_error[1][i + 10] = error[0]
    CL_error[1][i + 10] = error[1]

for i in range(5):
    error = MAE(fold2_wknn_normal[i])
    HL_error[1][i + 15] = error[0]
    CL_error[1][i + 15] = error[1]

# set3 is test data
fold3_knn = nearestNeighbour(set3, set1, set2, set4, set5)
fold3_wknn = weightedKnn(set3, set1, set2, set4, set5)
fold3_knn_normal = nearestNeighbour(n_set3, n_set1, n_set2, n_set4, n_set5)
fold3_wknn_normal = weightedKnn(n_set3, n_set1, n_set2, n_set4, n_set5)

# we iterate through 4 different for loops and add MAE to columns for 4 different variations.(since each variation has 5 k-value we have total of 20 cols)
for i in range(5):
    error = MAE(fold3_knn[i])
    HL_error[2][i] = error[0]
    CL_error[2][i] = error[1]

for i in range(5):
    error = MAE(fold3_wknn[i])
    HL_error[2][i + 5] = error[0]
    CL_error[2][i + 5] = error[1]

for i in range(5):
    error = MAE(fold3_knn_normal[i])
    HL_error[2][i + 10] = error[0]
    CL_error[2][i + 10] = error[1]

for i in range(5):
    error = MAE(fold3_wknn_normal[i])
    HL_error[2][i + 15] = error[0]
    CL_error[2][i + 15] = error[1]

# set4 is test data
fold4_knn = nearestNeighbour(set4, set1, set2, set3, set5)
fold4_wknn = weightedKnn(set4, set1, set2, set3, set5)
fold4_knn_normal = nearestNeighbour(n_set4, n_set1, n_set2, n_set3, n_set5)
fold4_wknn_normal = weightedKnn(n_set4, n_set1, n_set2, n_set3, n_set5)

# we iterate through 4 different for loops and add MAE to columns for 4 different variations.(since each variation has 5 k-value we have total of 20 cols)
for i in range(5):
    error = MAE(fold4_knn[i])
    HL_error[3][i] = error[0]
    CL_error[3][i] = error[1]

for i in range(5):
    error = MAE(fold4_wknn[i])
    HL_error[3][i + 5] = error[0]
    CL_error[3][i + 5] = error[1]

for i in range(5):
    error = MAE(fold4_knn_normal[i])
    HL_error[3][i + 10] = error[0]
    CL_error[3][i + 10] = error[1]

for i in range(5):
    error = MAE(fold4_wknn_normal[i])
    HL_error[3][i + 15] = error[0]
    CL_error[3][i + 15] = error[1]

# set5 is test data
fold5_knn = nearestNeighbour(set5, set1, set2, set3, set4)
fold5_wknn = weightedKnn(set5, set1, set2, set3, set4)
fold5_knn_normal = nearestNeighbour(n_set5, n_set1, n_set2, n_set3, n_set4)
fold5_wknn_normal = weightedKnn(n_set5, n_set1, n_set2, n_set3, n_set4)

# we iterate through 4 different for loops and add MAE to columns for 4 different variations.(since each variation has 5 k-value we have total of 20 cols)
for i in range(5):
    error = MAE(fold5_knn[i])
    HL_error[4][i] = error[0]
    CL_error[4][i] = error[1]

for i in range(5):
    error = MAE(fold5_wknn[i])
    HL_error[4][i + 5] = error[0]
    CL_error[4][i + 5] = error[1]

for i in range(5):
    error = MAE(fold5_knn_normal[i])
    HL_error[4][i + 10] = error[0]
    CL_error[4][i + 10] = error[1]

for i in range(5):
    error = MAE(fold5_wknn_normal[i])
    HL_error[4][i + 15] = error[0]
    CL_error[4][i + 15] = error[1]

pd.set_option('display.precision', 2)
HL_error = np.vstack([HL_error, np.mean(HL_error, axis = 0)])

HL_df = pd.DataFrame(HL_error, columns = ['k=1, knn, raw data', 'k=3, knn, raw data', 'k=5, knn, raw data', "k=7, knn, raw data",
                                         'k=9, knn, raw data', 'k=1, w-knn, raw data', 'k=3, w-knn, raw data', "k=5, w-knn, raw data",
                                         'k=7, w-knn, raw data', 'k=9, w-knn, raw data',
                                         'k=1, knn, normalized data', 'k=3, knn, normalized', 'k=5, knn, normalized', "k=7, knn, normalized",
                                         'k=9, knn, normalized', 'k=1, w-knn, normalized', 'k=3, w-knn, normalized', "k=5, w-knn, normalized",
                                         'k=7, w-knn, normalized', 'k=9, w-knn, normalized'])
HL_df.index = ["Fold1", "Fold2", "Fold3", "Fold4", "Fold5", "Average"]
#HL_df = HL_df.style.set_caption("Mean Absolute Error for Heating Load")
print(HL_df)

CL_error = np.vstack([CL_error, np.mean(CL_error, axis = 0)])

CL_df = pd.DataFrame(CL_error, columns = ['k=1, knn, raw data', 'k=3, knn, raw data', 'k=5, knn, raw data', "k=7, knn, raw data",
                                         'k=9, knn, raw data', 'k=1, w-knn, raw data', 'k=3, w-knn, raw data', "k=5, w-knn, raw data",
                                         'k=7, w-knn, raw data', 'k=9, w-knn, raw data',
                                         'k=1, knn, normalized data', 'k=3, knn, normalized', 'k=5, knn, normalized', "k=7, knn, normalized",
                                         'k=9, knn, normalized', 'k=1, w-knn, normalized', 'k=3, w-knn, normalized', "k=5, w-knn, normalized",
                                         'k=7, w-knn, normalized', 'k=9, w-knn, normalized'])
CL_df.index = ["Fold1", "Fold2", "Fold3", "Fold4", "Fold5", "Average"]
#CL_df = CL_df.style.set_caption("Mean Absolute Error for Cooling Load")
print(CL_df)

df.head(5)




