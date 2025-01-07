# -*- Encoding:UTF-8 -*-
#地区隐藏关键词
#测试数据为用户分为两组（每组两个）差评的10条数据均相同，每组好评的数据有20个相同，8个不同
import numpy as np
import sys

from scipy import io
import os
class DataSet(object):
    def __init__(self,user_num):
        self.user_num=user_num
        self.data,self.shape = self.getData()
        self.neg_data = self.getnegData()
        self.train, self.test = self.getTrainTest()
        self.trainDict = self.getTrainDict()
    def getData(self):#正例的
        # data_active = io.loadmat('Data/ml-1m/P2P1700/active_user.mat') # 读取active_id
        # data_act = data_active['active_user'].astype(int)[:, 0]#第0列
        # user_map = {}  # 必须重索引
        # for index, user_id in enumerate(data_act, start=1):
        #     user_map[user_id] = index
        data_act=[i for i in range(1,self.user_num+1)]
        user_map = {}
        for index, user_id in enumerate(data_act, start=1):
             user_map[user_id] = index
        #定义物品列
        data_available_item = io.loadmat(r'F:\python\learningproject\nlprecommend\Deep_Matrix_Factorization_Models-master-master\Deep_Matrix_Factorization_Models-master\Data\recommend_item.mat')  # 读取available_item_id
        data_available = data_available_item['recommend_item'].astype(int)[:, 0]
        item_map = {}  # 同上
        for index, item_id in enumerate(data_available, start=1):
             item_map[item_id] = index
        # 交互列表以dict形式保存
        user_to_item = {}
        # data = io.loadmat(path)  # 读取交互数据:交互过的政策id数组
        # for index, i in enumerate(data['U_I'], start=1):
        #     user_to_item[index] = i[0][0].tolist()  # 字典键值可以是列表{1:[itemid,item_id]}
        pos_data = io.loadmat(
            r'F:\python\learningproject\nlprecommend\Deep_Matrix_Factorization_Models-master-master\Deep_Matrix_Factorization_Models-master\Data\pos_data.mat')  # 读取active_id
        for index, i in enumerate(pos_data['pos_data'], start=1):
            user_to_item[index] = i[0][0].tolist()
        user_num = self.user_num
        item_num = len(data_available)# 544self.shape的大小[user_num,item_num]
        print(item_num)
        data = []  # 重新分配data内存[(user,item,score,time)]
        pad_user=[]
        for userid in range(1,user_num+1):#len(user_to_item)
            if userid not in data_act:  # 如果不在
                continue
            flag = 1  # 没有时间戳，所以只能以0-1-2代替
            for itemid in user_to_item[userid]:
                if itemid in data_available:  # 只保存活跃的item
                    data.append((user_map[userid], item_map[itemid], 1, flag))  # usr,item,1,time；从1开始标号
                flag = flag + 1
            else:
                pad_user.append(userid)
        self.maxRate = 1#正例的最大评分
        for i in pad_user:#用第一条政策作为填充
            data.append((i,1,1,1))
        return data, [user_num, item_num]  # user从0开始的
    #?:时间搓：日志？
    #填充

    def getnegData(self):#负例
        data_act = [i for i in range(1, self.user_num + 1)]
        user_map = {}
        for index, user_id in enumerate(data_act, start=1):
            user_map[user_id] = index
        data_available_item = io.loadmat(
            r'F:\python\learningproject\nlprecommend\Deep_Matrix_Factorization_Models-master-master\Deep_Matrix_Factorization_Models-master\Data\recommend_item.mat')  # 读取available_item_id
        data_available = data_available_item['recommend_item'].astype(int)[:, 0]
        item_map = {}  # 同上
        for index, item_id in enumerate(data_available, start=1):
            item_map[item_id] = index
        user_num = self.user_num
        neg_user_to_item = {}
        neg_data = io.loadmat(
            r'F:\python\learningproject\nlprecommend\Deep_Matrix_Factorization_Models-master-master\Deep_Matrix_Factorization_Models-master\Data\neg_data.mat')  # 读取active_id
        for index, i in enumerate(neg_data['neg_data'], start=1):
            neg_user_to_item[index] = i[0][0].tolist()
        neg_data = []  # 重新分配data内存[(user,item,score,time)]
        pad_user=[]
        for userid in range(1, user_num + 1):  # len(user_to_item)
            if userid not in data_act:  # 如果不在
                continue
            flag = 1  # 没有时间戳，所以只能以0-1-2代替
            for itemid in neg_user_to_item[userid]:
                if itemid in data_available:  # 只保存活跃的item
                    neg_data.append((user_map[userid], item_map[itemid], 0, flag))  # usr,item,1,time；从1开始标号
                flag = flag + 1
            else:
                pad_user.append(userid)
        self.minRate = 0  # 负例的最大评分
        for i in pad_user:  # 用最后一条政策作为填充
            neg_data.append((i, self.shape[1]-1, 1, 1))
        return neg_data

    def getTrainTest(self):#划分正例训练和测试集
        data = self.data # user, movie, score, time
        data = sorted(data, key=lambda x: (x[0], x[3]))#按用户和时间搓排序
        train = []
        test = []
        for i in range(len(data)-1):
            user = data[i][0]-1 # 这里选择都从0开始index
            item = data[i][1]-1
            rate = data[i][2]
            if data[i][0] != data[i+1][0]:
                test.append((user, item, rate)) # 每个用户留了一个项目用作测试，这个交互过的项目应该是最高的才对。
            else:
                train.append((user, item, rate))
        test.append((data[-1][0]-1, data[-1][1]-1, data[-1][2]))
        return train, test

    def getTrainDict(self):
        dataDict = {}
        for i in self.train:
            dataDict[(i[0], i[1])] = i[2]
        return dataDict#返回(用户，物品)：评分

    def getEmbedding(self):
        train_matrix = np.zeros([self.shape[0], self.shape[1]], dtype=np.float32)
        for i in self.train:
            user = i[0]
            movie = i[1]
            rating = i[2]
            train_matrix[user][movie] = rating # 训的时候是一个完整的矩阵，感觉是作为训练的label使用的一个查找表格
        return np.array(train_matrix)#会将每个用户嵌入所有政策矩阵大小=用户数*推荐物品总数，所以主要训练的是正例，

    def getInstances(self):#在原来正例但不是train数据集的基础上添加训练的负例
        neg_data = self.neg_data  # user, movie, score, time
        neg_data = sorted(neg_data, key=lambda x: (x[0], x[3]))
        train = self.train
        user = []
        item = []
        rate = []
        for i in train:
            user.append(i[0])
            item.append(i[1])
            rate.append(i[2])
        for i in range(len(neg_data) - 1):
            user.append(neg_data[i][0] - 1)  # 这里选择都从0开始index
            item.append(neg_data[i][1] - 1)
            rate.append(neg_data[i][2])
        return np.array(user), np.array(item), np.array(rate)

    def getTestNeg(self, testData):#重点给未看过的政策
        user = []
        item = []
        for s in testData: # s:(user, item, rate) 正例
            tmp_user = []
            tmp_item = []
            u = s[0]
            i = s[1]
            tmp_user.append(u) # [user_id]
            tmp_item.append(i) #[posId]
            neglist = set()
            neglist.add(i) # [posId]
            for t in range(self.shape[1]//77):#越多越准确,目前占比544/7=77
                j = np.random.randint(self.shape[1])
                while (u, j) in self.trainDict or j in neglist:#给未看过的政策
                    j = np.random.randint(self.shape[1])
                neglist.add(j)
                tmp_user.append(u) # [userId,userId,] 100个
                tmp_item.append(j) # # [posId,negId,...] # 100个
            user.append(tmp_user) # [[userID,...],[usrID2,...]]
            item.append(tmp_item) # [[posId,negId,...],[posId,negId,...]]
        #[46, 53, 57, 30, 27, 59]
        return [np.array(user), np.array(item)] # 这里是用于Test评测指标计算时候使用的neg item+ 正例
