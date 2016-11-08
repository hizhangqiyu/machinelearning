#!/usr/bin/env python

#  Author: AppleFairy
#  Date: 8/11/2016
#
#
# *************************************** #
from __future__ import division
from sklearn import metrics
import math

class FMeasureClustering:

    def __init__ (self, goldenLabels, predLabels):
        if len(goldenLabels) == len(predLabels):
            self.__classList = set(goldenLabels)
            self.__clusterList = set(predLabels)
            self.__goldenLabels = goldenLabels
            self.__predLabels = predLabels
        else:
            print("The number of golden and predict label must be equal to each other")
            exit(1)

    # beta=1 by default(F1-measure) 
    def __GenerateDataSet(self, name , label):
        labels_new = []
        for element in label:
            if element != name:
                labels_new.append(0)
            else:
                labels_new.append(1)
        return labels_new

    def __CalculateNum (self, name):
        classNum = 0;
        for elem in self.__goldenLabels:
            if elem == name:
                classNum = classNum + 1
        return classNum

    def __TruePositive(self, y_true, y_pred):
        tp = 0
        for idx in range(len(y_true)):
            if y_true[idx] == y_pred[idx] and y_true[idx] == 1:
                tp = tp +1
        return tp

    def __Precision(self, y_true, y_pred):
        tp = self.__TruePositive(y_true, y_pred)
        return tp/y_pred.count(1)

    def __Recall(self, y_true, y_pred):
        tp = self.__TruePositive(y_true, y_pred)
        return tp/y_true.count(1)

    def __FMeasure(self, y_true, y_pred, eps):
        precision = self.__Precision(y_true, y_pred)
        recall = self.__Recall(y_true, y_pred)
        if 0 == precision and 0 == recall:
            return 0
        else:
            return (math.pow(eps,2) +1)*(precision*recall)/(math.pow(eps,2)*precision+recall)

    def __Mapping(self, className, eps):
        maxPrecision = 0
        mappedClusterName = 0
        for clusterName in self.__clusterList:
            y_true = self.__GenerateDataSet(className, self.__goldenLabels)
            y_pred = self.__GenerateDataSet(clusterName, self.__predLabels)
            f = self.__FMeasure(y_true, y_pred, eps)

            if f > maxPrecision:
                maxPrecision = f
                mappedClusterName = clusterName

        print("map %d => %d" % (className, mappedClusterName))
        return [maxPrecision, mappedClusterName]

    def GetFMeasure(self, eps=1):
        result = 0
        for element in self.__classList:
            print("F-measure result{}:".format(element))
            f, n = self.__Mapping(element, eps)
            result = result + f * self.__goldenLabels.count(n)
        return result/len(self.__goldenLabels)

labels_golden = [0,0,0,0,1,2,2,2,1,3]
labels = [0,0,0,0,2,1,1,1,3,4]

print("AMI result:")
print metrics.adjusted_mutual_info_score(labels_golden, labels)

print("V-measure result:")
print metrics.v_measure_score(labels_golden, labels)

fmeasure = FMeasureClustering(labels_golden, labels)
print fmeasure.GetFMeasure()
