# -*- Encoding:UTF-8 -*-
import tensorflow as tf
import numpy as np
import pandas as pd
import argparse
from DataSet import DataSet
import sys
import os
import heapq
import math
tf.compat.v1.disable_eager_execution()#优化器可以直接计算损失，无需传入函数
def main():
    a=3
    b=2#外部传参
    parser = argparse.ArgumentParser(description="Options")
    #parser.add_argument('-dataName', action='store', dest='dataName', default=r'F:\python\learningproject\nlp\Deep_Matrix_Factorization_Models-master-master\Deep_Matrix_Factorization_Models-master\Data\ml-1m\P2P1700\UTOI_1700.mat')
    parser.add_argument('-userLayer', action='store', dest='userLayer', default=[512, 64])#用户层[512, 64]
    parser.add_argument('-itemLayer', action='store', dest='itemLayer', default=[1024, 64])#物品层[1024, 64]
    # parser.add_argument('-reg', action='store', dest='reg', default=1e-3)
    parser.add_argument('-topK', action='store', dest='topK', default=b)#默认前50个
    parser.add_argument('-actusrnum', action='store', dest='actusrnum', default=a)#外部传参有效用户数
    parser.add_argument('-lr', action='store', dest='lr', default=0.1)
    parser.add_argument('-maxEpochs', action='store', dest='maxEpochs', default=50, type=int)#最大批次
    parser.add_argument('-batchSize', action='store', dest='batchSize', default=256, type=int) #一批训练数
    parser.add_argument('-earlyStop', action='store', dest='earlyStop', default=30)
    parser.add_argument('-checkPoint', action='store', dest='checkPoint', default='./checkPoint/')#保存模型的参数
    args = parser.parse_args()
    classifier = Model(args)#进入模型
    save_recommend = classifier.run() #返回推荐字典{用户：推荐政策index}
    print(save_recommend)

class Model:
    def __init__(self, args):
        #顺序不要乱改
        #self.dataName = args.dataName
        self.actusrnum=args.actusrnum
        self.dataSet = DataSet(self.actusrnum)
        self.shape = self.dataSet.shape
        self.maxRate = self.dataSet.maxRate
        self.minRate = self.dataSet.minRate
        self.train = self.dataSet.train
        self.test = self.dataSet.test
        #self.negNum = args.negNum
        self.testNeg = self.dataSet.getTestNeg(self.test)
        self.add_embedding_matrix()
        self.add_placeholders()
        self.userLayer = args.userLayer
        self.itemLayer = args.itemLayer
        self.add_model()
        self.add_loss()
        self.lr = args.lr
        self.add_train_step()
        self.checkPoint = args.checkPoint
        self.init_sess()
        self.maxEpochs = args.maxEpochs
        self.batchSize = args.batchSize
        self.topK = args.topK
        self.earlyStop = args.earlyStop
    def add_placeholders(self):
        self.user = tf.compat.v1.placeholder(dtype=tf.int32)
        self.item = tf.compat.v1.placeholder(dtype=tf.int32)
        self.rate = tf.compat.v1.placeholder(dtype=tf.float32)
        self.drop = tf.compat.v1.placeholder(dtype=tf.float32)

    def add_embedding_matrix(self):
        self.user_item_embedding = tf.convert_to_tensor(self.dataSet.getEmbedding()) # 使用用户的初始交互记录作为embeddng初始化。随机初始化能跑通，但是不够
        self.item_user_embedding = tf.transpose(self.user_item_embedding) # 反转作为embedding

    def add_model(self):
        user_input = tf.nn.embedding_lookup(self.user_item_embedding, self.user) # batch_user_num*embedding_size
        item_input = tf.nn.embedding_lookup(self.item_user_embedding, self.item) # batch_item_num*embedding_size
        def init_variable(shape, name):
            return tf.Variable(tf.random.truncated_normal(shape=shape, dtype=tf.float32, stddev=0.01), name=name)
        with tf.name_scope("User_Layer"):
            user_W1 = init_variable([self.shape[1], self.userLayer[0]], "user_W1") # item_size * 512
            user_out = tf.matmul(user_input, user_W1)
            for i in range(0, len(self.userLayer)-1): # 两层0,1
                W = init_variable([self.userLayer[i], self.userLayer[i+1]], "user_W"+str(i+2))
                b = init_variable([self.userLayer[i+1]], "user_b"+str(i+2))
                user_out = tf.nn.relu(tf.add(tf.matmul(user_out, W), b)) # user_num*64
        with tf.name_scope("Item_Layer"):
            item_W1 = init_variable([self.shape[0], self.itemLayer[0]], "item_W1") # user_size * 1024
            item_out = tf.matmul(item_input, item_W1) # item_num*64
            for i in range(0, len(self.itemLayer)-1):
                W = init_variable([self.itemLayer[i], self.itemLayer[i+1]], "item_W"+str(i+2))
                b = init_variable([self.itemLayer[i+1]], "item_b"+str(i+2))
                item_out = tf.nn.relu(tf.add(tf.matmul(item_out, W), b)) # item_num * 64 relu有概率输出全0值
        norm_user_output = tf.sqrt(tf.reduce_sum(tf.square(user_out), axis=1)) # user_size
        norm_item_output = tf.sqrt(tf.reduce_sum(tf.square(item_out), axis=1)) # item_size
        # y_该是[user_num,]，这里没有使用softmax而是直接输出了user对某个item的值,其实就是判断为真则靠近1，反之靠近0
        self.y_ = tf.reduce_sum(tf.multiply(user_out, item_out), axis=1) / (norm_item_output * norm_user_output + 1e-5)
        self.y_ = tf.maximum(1e-6, self.y_) # 防止分母出现负数，计算的是得分，
        # shape可以随意变？
        # 即用户对于该item的偏好程度，如果这里是CTR,就是接上一个softmax作为二分类，最终test的时候会有差别，我们
        # 看的指标就是test中判断正确的情况比如AUC，就不用按照这个偏好进行排名了，需要遍历一个用户和多个item, 然后算总的。

    def add_loss(self):
        regRate = self.rate / self.maxRate
        losses = regRate * tf.math.log(self.y_) + (1 - regRate) * tf.math.log(1 - self.y_) # 当然了负样本评分肯定是0啊
        loss = -tf.reduce_sum(losses)
        # regLoss = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()])
        # self.loss = loss + self.reg * regLoss
        self.loss = loss
    def add_train_step(self):
        '''
        global_step = tf.Variable(0, name='global_step', trainable=False)
        self.lr = tf.train.exponential_decay(self.lr, global_step,
                                             self.decay_steps, self.decay_rate, staircase=True)
        '''
        # 旧版optimizer = tf.train.AdamOptimizer(self.lr),self.train_step = optimizer.minimize(self.loss)
        self.train_step=tf.compat.v1.train.GradientDescentOptimizer(learning_rate=self.lr).minimize(self.loss)
    #初始交互
    def init_sess(self):
        self.config = tf.compat.v1.ConfigProto()
        self.config.gpu_options.allow_growth = True
        self.config.allow_soft_placement = True
        self.sess = tf.compat.v1.Session(config=self.config)
        self.sess.run(tf.compat.v1.global_variables_initializer())
        self.saver = tf.compat.v1.train.Saver()
        if os.path.exists(self.checkPoint):
            # [os.remove(f) for f in os.listdir(self.checkPoint)]continue
            pass
        else:
            os.mkdir(self.checkPoint)#记录模型数据
    def run(self):
        #初始化
        best_hr = -1
        best_NDCG = -1
        best_epoch = -1
        print("Start Training!")
        for epoch in range(self.maxEpochs):
            print("="*20+"Epoch ", epoch, "="*20)
            self.run_epoch(self.sess)
            print('='*50)
            print("Start Evaluation!")
            hr, NDCG ,save_recommend = self.evaluate(self.sess, self.topK) # 这里可以返回保存我们的东西，每次都要评估一次，那我们每次就保存一个吧
            print("Epoch ", epoch, "HR: {}, NDCG: {}".format(hr, NDCG))
            # 当NDCG有一段时间不上升了就早停
            if hr > best_hr or NDCG > best_NDCG:
                best_hr = hr
                best_NDCG = NDCG
                best_epoch = epoch
                self.saver.save(self.sess, self.checkPoint)
            if epoch - best_epoch > self.earlyStop:
                print("Normal Early stop!")
                break
            print("="*20+"Epoch ", epoch, "End"+"="*20)
        print("Best hr: {}, NDCG: {}, At Epoch {}".format(best_hr, best_NDCG, best_epoch))
        print("Training complete!")
        return save_recommend # 返回我们需要的列表
    def run_epoch(self, sess, verbose=10):
        train_u, train_i, train_r = self.dataSet.getInstances()
        train_len = len(train_u)
        shuffled_idx = np.random.permutation(np.arange(train_len))
        train_u = train_u[shuffled_idx] # 打散ID
        train_i = train_i[shuffled_idx]
        train_r = train_r[shuffled_idx]
        num_batches=5#num_batches = len(train_u) // self.batchSize + 1
        losses = []
        for i in range(num_batches):
            min_idx = i * self.batchSize
            max_idx = np.min([train_len, (i+1)*self.batchSize])
            train_u_batch = train_u[min_idx: max_idx] # 每次随机选一堆user和item进行训练
            train_i_batch = train_i[min_idx: max_idx]
            train_r_batch = train_r[min_idx: max_idx]
            #评分为0/1
            feed_dict = self.create_feed_dict(train_u_batch, train_i_batch, train_r_batch)
            #sess.run：fetch列表/tensor，feed_dict以字典的方式填充空位
            _, tmp_loss = sess.run([self.train_step, self.loss], feed_dict=feed_dict) # 一个batch 256 每10个batch是一个step，把这10个step的loss输出
            losses.append(tmp_loss)
            if verbose and i % verbose == 0:
                sys.stdout.write('\r{} / {} : loss = {}'.format(i, num_batches, np.mean(losses[-verbose:])))
                sys.stdout.flush()
        loss = np.mean(losses)
        print("\nMean loss in this epoch is: {}".format(loss))
        return loss
    def create_feed_dict(self, u, i, r=None, drop=None):
        return {self.user: u,
                self.item: i,
                self.rate: r,
                self.drop: drop}
    #评估
    def evaluate(self, sess, topK):
        def getHitRatio(ranklist, targetItem):
            for item in ranklist:
                if item == targetItem:
                    return 1
            return 0
        def getNDCG(ranklist, targetItem):
            for i in range(len(ranklist)): # NDCG的计算公式
                item = ranklist[i]
                if item == targetItem:
                    return math.log(2) / math.log(i+2)
            return 0 # 如果groundtruth不在里面就为0
        hr =[]
        NDCG = []
        testUser = self.testNeg[0]  #[[userID,...],[usrID2,...]]
        testItem = self.testNeg[1] #[[posId,negId,...],[posId2,negId,...]]
        save_recommend = {} # 后添加的
        for i in range(len(testUser)): # 对每一个用户
            target = testItem[i][0] # 正例
            feed_dict = self.create_feed_dict(testUser[i], testItem[i])
            predict = sess.run(self.y_, feed_dict=feed_dict) # user_num * item_num
            item_score_dict = {} # 用户对于test-data的评分
            for j in range(len(testItem[i])): # 对于所有的物品
                item = testItem[i][j] # item_id
                item_score_dict[item] = predict[j]  # {item1:0.2,item3:0.3,......}, 即[0.2,0.3,0.1]等，长度就是正例+负例的数量
            ranklist = heapq.nlargest(topK, item_score_dict, key=item_score_dict.get) # 给每个user推荐排名最高的topK个item id，排序后的
            # 我们如果需要保留数据可以在这里加
            # 保存列表
            save_recommend[testUser[i][0]] = np.array(ranklist)
            tmp_hr = getHitRatio(ranklist, target)
            tmp_NDCG = getNDCG(ranklist, target)
            hr.append(tmp_hr)
            NDCG.append(tmp_NDCG)
        return np.mean(hr), np.mean(NDCG),save_recommend

if __name__ == '__main__':
    main()
