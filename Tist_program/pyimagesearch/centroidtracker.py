from scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np
import cv2

class CentroidTracker:
    def __init__(self, maxDisappeared=10, maxDistance=20):
        self.nextObjectID = 0
        #編號
        self.objects = OrderedDict()
        #儲存編號與座標
        self.disappeared = OrderedDict()
        #目標消失的偵數
        self.maxDisappeared = maxDisappeared
        self.maxDistance = maxDistance

    def register(self, centroid):
        #新的目標加入到字典
        self.objects[self.nextObjectID] = centroid
        self.disappeared[self.nextObjectID] = 0
        self.nextObjectID += 1
        
    #目標消失，删除字典裡的目標
    def deregister(self, objectID):
        del self.objects[objectID]
        del self.disappeared[objectID]

    #最重要的一步，更新目標，即追踪目標
    def update(self, rects, frame):
        #首先判斷，如果沒有框，則將消失的偵數加一，超過設置值，註銷目標。
        # print(len(rects))
        if len(rects) == 0:
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)   
            return self.objects

        inputCentroids = np.zeros((len(rects), 2), dtype="int")


        for (i, (startX, startY, endX, endY)) in enumerate(rects):
            # print((startX, startY, endX, endY))
            # print((startX + endX) / 2.0)
            cX = int((startX + endX) / 2.0)#換算最接近的整數
            cY = int((startY + endY) / 2.0)
            if cX < 1:
                cX = 0
            if cY < 1:
                cY = 0
            try:
                inputCentroids[i] = (cX, cY)
            except:
                inputCentroids[i] = (0, 0)
            
            # print(inputCentroids[i])
            
            #將這一偵目標個數的中心點加入字典
            #截下圖片
            # img = frame[int(startY):int(endY), int(startX):int(endX)]
            # cv2.imwrite('frame/people_{}.png'.format(i), img)

        #若未追蹤任何目標，則加入追蹤
        if len(self.objects) == 0:
            for i in range(0, len(inputCentroids)):
                self.register(inputCentroids[i])

        else:
            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())

            #計算目標的兩個集合的歐式距離
            D = dist.cdist(np.array(objectCentroids), inputCentroids)

            rows = D.min(axis=1).argsort()
            #找出美行最小值，並求出奇索引，返回列表

            cols = D.argmin(axis=1)[rows]
            #找出最近的目標

            usedRows = set()
            usedCols = set()

            # loop over the combination of the (row, column) index
            # tuples
            for (row, col) in zip(rows, cols):
                #判斷目標是否已經分配ID
                if row in usedRows or col in usedCols:
                    continue

                #設置最大距離
                if D[row, col] > self.maxDistance:
                    continue

                #分配ID，並將該ID對應中心點賦予原來的目標，更新目標點
                objectID = objectIDs[row]
                self.objects[objectID] = inputCentroids[col]
                self.disappeared[objectID] = 0

                #尋找未被分配的目標
                usedRows.add(row)
                usedCols.add(col)

            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)

            #判斷目標是消失還是增加，如果消失，計算消失偵數，如果增加，加入一個新的目標點
            if D.shape[0] >= D.shape[1]:
                for row in unusedRows:
                    objectID = objectIDs[row]
                    self.disappeared[objectID] += 1

                    if self.disappeared[objectID] > self.maxDisappeared:
                        self.deregister(objectID)

            else:
                for col in unusedCols:
                    self.register(inputCentroids[col])

        return self.objects
    
    
