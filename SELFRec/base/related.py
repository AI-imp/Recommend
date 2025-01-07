import csv
import pandas as pd
import math
import json
from collections import Counter
import os
from base.database import database


# 获取相关用户
class relate_user:
    def __init__(self, arg, dataset=None, test=0):

        self.Member_dir = arg['member']
        self.MeetingMember_dir = arg['meetingmember']
        self.ActivityAttendance_dir = arg['activityattendance']
        self.hisdir = arg['historydir']
        self.excel_file = arg['excel_file']
        self.savedir = arg['savedir']
        self.read_for_dataset = self.return_bool(arg['read_for_dataset'])
        if not test:
            self.save_json = os.path.join(self.savedir, "sort.json")

        if self.read_for_dataset:
            self.datasetconf = dataset
            self.use_proxy = self.return_bool(dataset['proxy'])
            self._chang_dir(arg)
            self.read_data_dataset(self.Member_dir)

    def _chang_dir(self, arg):
        self.Member_dir = eval(arg['dataset_member'])
        self.MeetingMember_dir = eval(arg['dataset_meetingmember'])
        self.ActivityAttendance_dir = eval(arg['dataset_activityattendance'])
        self.excel_file = eval(arg['dataset_excel_file'])

    def his(self,df=None):
        '''
        通过用户交互历史的方式找到相似用户
        '''
        df_userhis = self.process_page_history(df)
        # print(df_userhis)
        df_temp = self.process_data()
        df_userhis['PagePath'] = df_userhis['PagePath'].str.extract('(\d+)').astype(int)
        # 使用merge方法合并DataFrame，基于PagePath和Id进行匹配
        merged_df = df_userhis.merge(df_temp, left_on='PagePath', right_on='Id', how='left')
        # 覆盖PagePath的值为CreatedBy的值
        merged_df['PagePath'] = merged_df['CreatedBy']

        # 删除不再需要的列
        merged_df.drop(columns=['Id', 'CreatedBy'], inplace=True)

        user_to_visited = {}
        for index, row in merged_df.iterrows():
            user, visited = row['PagePath'], row['Person']
            if user not in user_to_visited:
                user_to_visited[user] = Counter()
            user_to_visited[user][visited] += 1

        relate_his = {}
        for user, visited_counts in user_to_visited.items():
            relate_his[user] = []
            top_visited_users = visited_counts.most_common(6)
            for visited_user, count in top_visited_users:
                if int(user) != visited_user:
                    relate_his[user].append(visited_user)

        return relate_his

    def read_data_file(self, file):
        return pd.read_csv(file)

    def read_data_dataset(self, dataset):
        # print(dataset)
        db = self.connect_dabase(dataset[0])
        df = db.readTable_pd(dataset[1])
        # print(df)
        db.disconnect()
        return df

    def return_df(self,dir):
        if self.read_for_dataset:
            return self.read_data_dataset(dir)
        else:
            return self.read_data_file(dir)

    def process_data(self):
        # 读取Excel文件
        df = self.return_df(self.excel_file)

        # 提取PagePath列中的TopicId和Person
        df['Id'] = df['Id']
        df['CreatedBy'] = df['CreatedBy']
        new_df = pd.DataFrame({
            "CreatedBy": df['CreatedBy'],
            "Id": df['Id']
        })

        return new_df

    def meeting(self):
        '''
        通过会议的方式找到相似用户
        '''
        # 会议(以委员为中心)
        # Member
        df = self.return_df(self.Member_dir)
        df.fillna('others', inplace=True)
        id = df['Id']
        Name = df['Name']
        IsDeleted = df['IsDeleted']
        ###print(id)
        # MeetingMember
        df = self.return_df(self.MeetingMember_dir)
        df.fillna('others', inplace=True)
        MeetingId = df['MeetingId']
        MemberId = df['MemberId']
        s1 = {}
        for i in range(len(id)):
            if IsDeleted[i] == 0:
                s1[id[i]] = []
                for j in range(len(MemberId)):
                    if id[i] == MemberId[j]:
                        df_info = df[df['Id'] == j + 1]
                        if df[df['Id'] == j + 1].empty:
                            continue
                        else:
                            if df_info['IsDeleted'].values[0] == 0:
                                s1[id[i]].append(df_info['MeetingId'].values[0])

        intersection = None
        l = {}
        for k1, v1 in s1.items():
            l[k1] = []
            for v2 in s1.values():
                intersection = set(v1) & set(v2)
                intersection_list = len(list(intersection))
                l[k1].append(intersection_list)
        print(l[1])
        mm1 = {}
        for i, j in l.items():
            mm1[i] = []
            max_values, max_indices = self.get_max_elements(j)
            max_values = [value for value, index in zip(max_values, max_indices) if index != i]
            max_indices = [index for index in max_indices if index != i]
            for value, index in zip(max_values, max_indices):
                mm1[i].append([index, value])
        print(mm1)

        mm2 = {}
        for i, j in l.items():
            mm2[i] = []
            max_values, max_indices = self.get_max_elements(j)
            max_values = [value for value, index in zip(max_values, max_indices) if index != i]
            max_indices = [index for index in max_indices if index != i]
            for value, index in zip(max_values, max_indices):
                mm2[i].append([index, value])
        print(mm2)

        # Member
        df = self.return_df(self.Member_dir)
        df.fillna('others', inplace=True)
        id = df['Id']
        Name = df['Name']
        IsDeleted = df['IsDeleted']
        ###print(id)
        # MeetingMember
        df = self.return_df(self.ActivityAttendance_dir)
        df.fillna('others', inplace=True)
        ActivityId = df['ActivityId']
        MemberId = df['MemberId']
        s1 = {}
        for i in range(len(id)):
            if IsDeleted[i] == 0:
                s1[id[i]] = []
                for j in range(len(MemberId)):
                    if id[i] == MemberId[j]:
                        df_info = df.loc[j]
                        if df.loc[j].empty:
                            continue
                        else:
                            s1[id[i]].append(df_info['ActivityId'])
        print(s1)

        intersection = None
        l = {}
        for k1, v1 in s1.items():
            l[k1] = []
            for v2 in s1.values():
                intersection = set(v1) & set(v2)
                intersection_list = len(list(intersection))
                l[k1].append(intersection_list)
        print(l[1])

        mm3 = {}
        for i, j in l.items():
            mm3[i] = []
            max_values, max_indices = self.get_max_elements(j)
            max_values = [value for value, index in zip(max_values, max_indices) if index != i]
            max_indices = [index for index in max_indices if index != i]
            for value, index in zip(max_values, max_indices):
                mm3[i].append([index, value])
        print(mm3)

        a = mm1
        b = mm2
        c = mm3
        d = {}
        p = 1
        keys = list(a.keys())
        d = {}
        for i in range(len(keys)):
            d[i + 1] = []
            try:
                x = a[keys[i]]
                y = b[keys[i]]
            except KeyError:
                continue
            for j in x:
                d[i + 1].append([j[0], j[1]])
            for k in y:
                d[i + 1].append([k[0], k[1]])
        print(d)

        e = {}
        keys = list(d.keys())
        for i in range(len(keys)):
            e[i + 1] = []
            try:
                x = c[keys[i]]
                y = d[keys[i]]
            except KeyError:
                continue
            for j in x:
                e[i + 1].append([j[0], j[1]])
            for k in y:
                e[i + 1].append([k[0], k[1]])
        print(e)

        sum_dict = {}

        sort_dict = {}
        for i in range(1, len(e) + 1):

            for item in e[i]:
                key = item[0]
                value = item[1]
                if key in sum_dict:
                    sum_dict[key] += value
                else:
                    sum_dict[key] = value
            sorted_items = sorted(sum_dict.items(), key=lambda x: x[1], reverse=True)
            top_six = dict(sorted_items[:6])
            # print(i) 
            wyid = []
            glcs = []

            for k, v in top_six.items():
                wyid.append(k)
                glcs.append(v)

            sort_dict[i] = wyid

        print(sort_dict)
        self.sort_dict = sort_dict
        return sort_dict

    def get_max_elements(self, lst):
        indexed_list = list(enumerate(lst))  # 创建索引和元素组成的元组列表
        sorted_list = sorted(indexed_list, key=lambda x: x[1], reverse=True)  # 根据元素值进行降序排序
        max_elements = sorted_list[:10]  # 获取排序后的前10个元素

        # 提取最大元素和索引
        max_values = [element for index, element in max_elements]
        max_indices = [index + 1 for index, element in max_elements]

        return max_values, max_indices

    def process_page_history(self,his_df=None):
        # 读取excel文件
        if not his_df.empty:
            df = his_df
        else:
            df = pd.read_excel(self.hisdir)
        # 提取PagePath列中ProID后的数值
        df['PagePath'] = df['PagePath'].str.extract('TopicId=(\d+)', expand=False)

        # 删除PagePath为空的行
        df = df.dropna(subset=['PagePath'])

        # 保留Person列
        df = df[['Person', 'PagePath']]
        df = df.drop_duplicates()

        # 返回结果
        return df

    def save_json(self, dir):
        with open(dir, 'w') as f:
            json.dump(self.sort_dict, f)

    # 将str转化为bool
    def return_bool(self, str_data):
        if str_data == "False":
            return False
        else:
            return True

    def connect_dabase(self, Database_name):
        conf = self.datasetconf
        use_proxy = self.use_proxy
        if use_proxy:
            db = database(conf['host'], conf['user'], conf['password'],
                          conf['port'], Database_name, sshset=conf['sshset']
                          )
        else:
            db = database(conf['host'], conf['user'], conf['password'],
                          conf['port'], Database_name)
        return db


if __name__ == "__main__":
    arg = {
        'member': r'C:\Users\12563\Desktop\coda\dataset/priData/基础模块/Member.csv',
        'meetingmember': r'C:\Users\12563\Desktop\coda\dataset/priData/会议/MeetingMember.csv',
        "activityattendance": r'C:\Users\12563\Desktop\coda\dataset/priData/履职活动/ActivityAttendance.csv',
        'historydir': r"C:\Users\12563\Desktop\coda\dataset/priData/PageHistory.xlsx",
        "excel_file": r"C:\Users\12563\Desktop\coda\dataset/priData/履职圈/Topic.csv",
        'savedir': r'C:\Users\12563\Desktop\coda\SELFRec / dataset / topic'
    }

    re = relate_user(arg)
