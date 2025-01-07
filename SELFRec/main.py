from util.settingconf import SettingConf
from base.database import database
from base.related import relate_user
from base.PageHistoryProcessor import PageHistoryProcessor
import os
from SELFRec import SELFRec
from util.conf import ModelConf
from base.get_rec_list import data_process
import ast
import json
import pandas as pd
from run.word_embedding import word2vec_process
import threading
from tqdm import tqdm
import time
import sys
import numpy as np


class run_project(object):
    def __init__(self, code_dir, Database_name, real_Database,table_name):
        self.code_dir = code_dir
        self.setting_file = os.path.join(code_dir, 'setting', 'setting.conf')
        self.conf_all = SettingConf(self.setting_file).config
        self.model = self.conf_all['userhistory']['model']
        self.use_proxy = self.return_bool(self.conf_all['database']['proxy'])
        self.Database_name = Database_name
        self.real_Database=real_Database
        self.table_name = table_name
        self.model_dir = self.conf_all['world2vec']['model_dir']
        self.his_df = None
        self.word2vec_model = None

    # 将str转化为bool
    def return_bool(self, str_data):
        if str_data == "False":
            return False
        else:
            return True

    def get_his_df(self):
        file = self.conf_all['userhistory']['dataset_historydir']
        file = ast.literal_eval(file)
        cnx = self.connect_dabase(self.real_Database)
        print('读取dataset_historydir 速度巨慢请耐心等待')
        df = cnx.readTable_pd(file[1])
        self.his_df = df

    def set_database(self, Database_name):
        self.Database_name = Database_name

    def set_table(self, table_name):
        self.table_name = table_name

    def connect_dabase(self, Database_name):
        conf = self.conf_all['database']
        use_proxy = self.use_proxy
        if use_proxy:
            db = database(conf['host'], conf['user'], conf['password'],
                          conf['port'], Database_name, sshset=conf['sshset']
                          )
        else:
            db = database(conf['host'], conf['user'], conf['password'],
                          conf['port'], Database_name)
        return db

    # 更新数据库
    def update_database_table(self, real_database,Database_name, table_name, new_data=None):
        '''
        Database_name:数据库名
        table_name：表名
        new_data：数据
        '''
        conf = self.conf_all['database']
        use_proxy = self.use_proxy
        if use_proxy:
            db = database(conf['host'], conf['user'], conf['password'],
                          conf['port'], real_database, sshset=conf['sshset'],
                          arg=self.conf_all[Database_name])
        else:
            db = database(conf['host'], conf['user'], conf['password'],
                          conf['port'], real_database, arg=self.conf_all[Database_name])

        # 有数据才更新
        if new_data != None:
            # 清空后写入
            print('清空表格')
            db.deleteTable(table_name)
            print("清空表格成功，开始写入")
            for user_id, rec_id in tqdm(new_data.items(), desc="Processing"):
                db.writeTable(table_name, ast.literal_eval(self.conf_all[Database_name]['table'])[table_name],
                              (user_id, str(rec_id)))
            print("写入成功")
        # data = db.readTable(table_name)
        print(new_data)
        db.disconnect()

    def data_cleaning(self):
        read_for_dataset = self.return_bool(self.conf_all['userhistory']['read_for_dataset'])

        file = self.conf_all['userhistory']['historydir']
        conf = self.conf_all[self.Database_name]
        processor = PageHistoryProcessor(file, conf, read_for_dataset)

        if read_for_dataset == True:
            if self.his_df:
                result = processor.process_page_history2(self.his_df)
            else:
                self.get_his_df()
                result = processor.process_page_history2(self.his_df)
        else:
            result = processor.process_page_history()

        dir = os.path.join(conf['savedir'], 'data.txt')
        processor.save_to_txt(result, dir)

    def model_tran(self):
        conf_dir = os.path.join(self.code_dir, 'conf', self.model + '.conf')
        print("Module config dir : ", conf_dir)
        conf = ModelConf(conf_dir)
        conf.config['training.set'] = os.path.join(self.conf_all[self.Database_name]['savedir'], 'data.txt')
        conf.config['test.set'] = os.path.join(self.conf_all[self.Database_name]['savedir'], 'data.txt')
        print('开始模型训练')
        rec = SELFRec(conf)
        print('模型训练结束')
        rec.execute()


    # 交换data中前两列数据
    def change_data(self):
        dir = os.path.join(self.conf_all[self.Database_name]['savedir'],'data.txt')
        df = pd.read_csv(dir, delimiter=' ', header=None)
        df[[0, 1]] = df[[1, 0]]
        df.to_csv(dir, sep=' ', index=False, header=False)




    def full_update(self):
        conf = self.conf_all
        database_name = self.Database_name
        need_clean = self.return_bool(conf[database_name]['need_clean'])
        dir = conf[database_name]['savedir']

        if need_clean == True:
            print("开始数据清洗")
            # 1. 数据清洗
            self.data_cleaning()
            print("数据清洗结束")

        # 2. 模型训练
        self.model_tran()

        # 3. 根据模型生成推荐结果
        print("开始根据模型生成推荐结果")
        dp = data_process(dir, 'user', 'item', conf[database_name]['k'], 'data')

        # 可以不保存
        dp.save(dir)
        new_data = dp.load_rec_list(dir)
        print("根据模型生成推荐结果结束")

        # self.table_name = eval(self.conf_all[self.Database_name]["table"])
        # # 4. 更新数据库
        self.update_database_table(self.real_Database,self.Database_name, self.table_name, new_data)

    def get_relate_user(self):
        conf = self.conf_all['T_topicRecommended']
        Related = relate_user(conf, dataset=self.conf_all['database'])
        sort_dict = Related.meeting()
        if self.his_df == None:
            self.get_his_df()
        sort_his = Related.his(self.his_df)
        dir = os.path.join(conf['savedir'], 'sort_dict.json')
        self.write_json(dir, sort_dict)
        dir = os.path.join(conf['savedir'], 'sort_his.json')

        sort_his = {str(key): str(value) for key, value in sort_his.items()}
        self.write_json(dir, sort_his)
        return sort_dict, sort_his

    # 将字典保存为json
    def write_json(self, file, data):
        with open(file, "w") as json_file:
            json.dump(data, json_file)

    # 读取
    def read_json(self, file):
        with open(file, "r", encoding='utf-8') as json_file:
            loaded_data = json.load(json_file)
        return loaded_data

    # 话题更新
    def topic_update(self, test=False):
        conf = self.conf_all
        if test:
            dir = os.path.join(conf['T_topicRecommended']['savedir'], 'sort_dict.json')
            sort_dict = self.read_json(dir)
            dir = os.path.join(conf['T_topicRecommended']['savedir'], 'sort_his.json')
            sort_his = self.read_json(dir)

        else:
            sort_dict, sort_his = self.get_relate_user()

        self.update_database_table(self.real_Database,self.Database_name, 'related_user', sort_dict)
        self.update_database_table(self.real_Database, self.Database_name,'related_user_his', sort_his)
        print(sort_dict)

    # 用于测试
    def get_topic_data_file(self):
        file = '/home/thr2/priData/履职圈/Topic.csv'
        return pd.read_csv(file)

    def get_tian_data_file(self):
        file = '/home/thr2/priData/提案/Propose.csv'
        return pd.read_csv(file)

    def get_tian_data(self):
        database = eval(self.conf_all['world2vec']['tian'])[0]
        table = eval(self.conf_all['world2vec']['tian'])[1]
        db = self.connect_dabase('rawdata')
        print('get_tian_data')
        df = db.readTable_pd(table)
        print('get_tian_data success')
        db.disconnect()
        return df

    # 通过数据库读取话题内容
    def get_topic_data(self):
        database = eval(self.conf_all['world2vec']['topic'])[0]
        table = eval(self.conf_all['world2vec']['topic'])[1]
        db = self.connect_dabase('AI')
        df = db.readTable_pd(table)
        db.disconnect()
        # print(df)
        return df

    def tian_classification(self):
        # 需要模型时加载提高速度
        if self.word2vec_model == None:
            print('word2vec_model 模型加载中 速度较慢请耐心等待')
            self.word2vec_model = word2vec_process(self.model_dir)
            print('word2vec_model 加载成功')
        df = self.get_tian_data()
        df = df[['Id', 'ContentAdvice']]

        # 删除 'ContentAdvice' 列中包含空白值的行
        df = df.dropna(subset=['ContentAdvice'])
        # 删除 'ContentAdvice' 列中类型为 float 的行
        df = df[~df['ContentAdvice'].apply(lambda x: isinstance(x, float))]
        # 删除 'ContentAdvice' 列中长度小于 10 的行
        df = df[df['ContentAdvice'].str.len() >= 10]

        file_dir = self.conf_all['world2vec']['label_dir']
        lable = self.read_json(file_dir)
        primary = int(self.conf_all['world2vec']['primary'])
        pri_lable = list(lable.keys())
        secondary = int(self.conf_all['world2vec']['secondary'])
        sub_lable = [element for sublist in list(lable.values()) for element in sublist]
        print('提案分类中')
        for index, row in tqdm(df.iterrows(), total=len(df)):
            # sentence_vector = self.word2vec_model.sentence_vector(row[1])
            # pri_topk_value = self.word2vec_model.topk_relation(sentence_vector, pri_lable, primary)
            # sub_topk_lable = self.word2vec_model.topk_relation(sentence_vector, sub_lable, secondary)
            pri_topk_label = self.word2vec_model.topk_relation(row[1], pri_lable, primary)
            sub_topk_lable = self.word2vec_model.topk_relation(row[1], sub_lable, secondary)
            df.at[index, 'pri_label'] = str(pri_topk_label)
            df.at[index, 'sub_lable'] = str(sub_topk_lable)

        file = os.path.join(self.conf_all['world2vec']['savedir'], 'tian', 'tian_label.csv')
        print('将提案更新到数据库中')
        df.to_csv(file, index=False, encoding='utf-8')

    def topic_classification(self):
        df = self.get_topic_data_file()
        df = df[['ProID', 'ContentAdvice']]

        # 删除 'ContentAdvice' 列中包含空白值的行
        df = df.dropna(subset=['ContentAdvice'])
        # 删除 'ContentAdvice' 列中类型为 float 的行
        df = df[~df['ContentAdvice'].apply(lambda x: isinstance(x, float))]
        # 删除 'ContentAdvice' 列中长度小于 10 的行
        df = df[df['ContentAdvice'].str.len() >= 10]

        file_dir = self.conf_all['world2vec']['label_dir']
        lable = self.read_json(file_dir)
        primary = int(self.conf_all['world2vec']['primary'])
        pri_lable = list(lable.keys())
        secondary = int(self.conf_all['world2vec']['secondary'])
        sub_lable = [element for sublist in list(lable.values()) for element in sublist]

        for index, row in tqdm(df.iterrows(), total=len(df)):
            # sentence_vector = self.word2vec_model.sentence_vector(row[1])
            # pri_topk_value = self.word2vec_model.topk_relation(sentence_vector, pri_lable, primary)
            # sub_topk_lable = self.word2vec_model.topk_relation(sentence_vector, sub_lable, secondary)
            pri_topk_label = self.word2vec_model.topk_relation(row[1], pri_lable, primary)
            sub_topk_lable = self.word2vec_model.topk_relation(row[1], sub_lable, secondary)
            df.at[index, 'pri_label'] = str(pri_topk_label)
            df.at[index, 'sub_lable'] = str(sub_topk_lable)

        file = os.path.join(self.conf_all['world2vec']['savedir'], 'tian', 'tian_label.csv')
        df.to_csv(file, index=False, encoding='utf-8')

    # 将csv文件保存到数据库中
    def csv_database(self, csv_file, dataset):
        df = pd.read_csv(csv_file)
        for key, value in dataset.items():
            database = key
            table_dict = value

        
        for key, value in table_dict.items():
            table = key
            tittle = value

        db = self.connect_dabase('AI')

        for index, row in tqdm(df.iterrows(), total=len(df)):
            values = tuple(row.values)
            str_data = tuple(str(x) for x in values)
            db.writeTable(table, tittle, str_data)

    def world2vec_updata(self):
        #############################################目前只支持提案分类
        # 提案分类
        self.tian_classification()
        # 将数据更新到数据库中
        file = os.path.join(self.conf_all['world2vec']['savedir'], 'tian', 'tian_label.csv')
        input_str = self.conf_all['world2vec']['tian_savedir']

        # 删除字符串中的单引号，以使其成为有效的JSON
        valid_json_str = input_str.replace("'", "\"")
        # 使用json.loads()将字符串转换为字典
        database_set = json.loads(valid_json_str)


        self.csv_database(file, database_set)

    def test(self):
        self.tian_classification()
        # file = os.path.join(self.conf_all['world2vec']['savedir'], 'tian', 'tian_label.csv')
        # input_str = self.conf_all['world2vec']['tian_savedir']
        #
        # # 删除字符串中的单引号，以使其成为有效的JSON
        # valid_json_str = input_str.replace("'", "\"")
        # # 使用json.loads()将字符串转换为字典
        # database_set = json.loads(valid_json_str)
        #
        # self.csv_database(file, database_set)


if __name__ == '__main__':
    setting_file = ''
    run = run_project('.', "T_topicRecommended", "AI",'user_to_tian')#第三个参数是topic的表

    #Activity 更新
    run.set_database('T_ActivityToMember')
    run.set_table("MemberToActivity")
    run.full_update()
    ###
    run.set_table("ActivityToMember")
    run.change_data()
    run.full_update()


    # tian更新
    run.set_database('T_tianRecommended')
    run.set_table("user_to_tian")
    run.full_update()
    ####
    run.set_table("tian_to_user")
    run.change_data()
    run.full_update()

    # 词向量训练
    run.world2vec_updata()

    # topic 更新
    run.set_database('T_topicRecommended')
    run.topic_update()

