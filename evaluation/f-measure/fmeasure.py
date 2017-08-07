#!/usr/bin/env python

#  Author: AppleFairy
#  Date: 8/11/2016
#
#
# *************************************** #

from __future__ import division
import pandas as pd
import math

class FMeasure:
    '''
        p : precision
        r : recall
        f : f measure
        
        tp: true positive
        fp: false positive
        fn: false negetive
        tn: true negetive
        
        p = tp / (tp + fp)
        r = tp / (tp + fn)
        f = (e^2 + 1) * (p * r) / (e^2 * p + r)
    '''
    def __init__(self, golden_vs_predict_result):
        self.__classIdList = set(golden_vs_predict_result.loc[:, "ClassId"].tolist())
        self.__clusterIdList = set(golden_vs_predict_result.loc[:, "ClusterId"].tolist())
        self.__gvpr = golden_vs_predict_result


    def __calculate(self, classId, clusterId, eps):
        truePositive = len(self.__gvpr.loc[(self.__gvpr.ClassId == classId) & (self.__gvpr.ClusterId == clusterId)])
        precision = truePositive/len(self.__gvpr[self.__gvpr.ClusterId == clusterId])
        recall = truePositive/len(self.__gvpr[self.__gvpr.ClassId == classId])

        if 0 == precision and 0 == recall:
            return 0
        else:
            return (math.pow(eps,2) +1)*(precision*recall)/(math.pow(eps,2)*precision+recall)

    def __mapping(self, classId, eps):
        maxPrecision = 0
        mappedClusterId = -1
        for clusterId in self.__clusterIdList:
            if(clusterId != -1):
                f = self.__calculate(classId, clusterId, eps)

                if f > maxPrecision:
                    maxPrecision = f
                    mappedClusterId = clusterId

        print(" class <==> cluster : %r <==> %r, precision=%f" % (classId, mappedClusterId, maxPrecision))
        return maxPrecision

    def get_fmeasure(self, eps=1):
        result = 0
        for classId in self.__classIdList:
            f = self.__mapping(classId, eps)
            result = result + f * len(self.__gvpr[self.__gvpr.ClassId == classId])

        return result/len(self.__gvpr)


if __name__ == "__main__":
    result = FMeasure([1,1,0,0,1,2],[0,0,1,1,0,2])
    print(result.get_fmeasure())