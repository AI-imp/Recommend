import os
import json
import torch
import numpy as np
import pickle


class data_process:
    def __init__(self, file_path, user_file_name, item_file_name, k, history, reverse=False) -> None:
        """
        :param file_path:文件名称
        :param user_file_name: 文件名字不包含后缀名（要求json 和 pt 文件名字相同后缀不同）
        :param item_file_name: 文件名字不包含后缀名（要求json 和 pt 文件名字相同后缀不同）
        :param history: 用户交互历史（要求json 和 pt 文件名字相同后缀不同）
        """
        self.reverse = reverse
        self.file_path = file_path
        self.user_file_name = user_file_name
        self.item_file_name = item_file_name
        self.k = k
        self.user_emb_path = os.path.join(self.file_path, self.user_file_name + ".pt")
        self.item_emb_path = os.path.join(self.file_path, self.item_file_name + ".pt")
        self.user_json_path = os.path.join(self.file_path, self.user_file_name + ".json")
        self.item_json_path = os.path.join(self.file_path, self.item_file_name + ".json")
        if history != None:
            file = os.path.join(self.file_path, history + ".txt")
            self.u_i_history = self.history(file)
        self.get_score()

    def get_score(self):
        """
        :return: 按照兴趣排序的字典前k个(不包含用户历史)
        """
        with open(self.user_json_path, 'r') as f:
            self.user = json.load(f)
        with open(self.item_json_path, 'r') as f:
            self.item = json.load(f)
        self.user_emb = torch.load(self.user_emb_path).cpu()
        self.item_emb = torch.load(self.item_emb_path).cpu()

        self.Interest_sorting = {}
        for i in self.user:
            row_index = torch.argsort(self.predict(i), descending=True)
            rec_item = []
            flag = 0
            for j in row_index:
                temp = self.get_item(j)
                if temp not in self.u_i_history[i]:
                    rec_item.append(temp)
                    flag += 1
                if flag == self.k:
                    break

            self.Interest_sorting[i] = rec_item
        return self.Interest_sorting

    def get_item(self, id):
        for key, value in self.item.items():
            if value == id:
                return key

    def get_user_id(self, u):
        if u in self.user:
            return self.user[u]

    def get_item_id(self, i):
        if i in self.item:
            return self.item[i]

    def predict(self, u):
        u = self.get_user_id(u)
        score = torch.matmul(self.user_emb[u], self.item_emb.transpose(0, 1))
        return score

    def history(self, file):
        u_i = {}
        with open(file, "r") as f:
            if self.reverse:
                for line in f:
                    line = line.split(" ")
                    if line[1] not in u_i.keys():
                        u_i[line[1]] = [line[0]]
                    else:
                        u_i[line[1]].append(line[0])
            else:
                for line in f:
                    line = line.split(" ")
                    if line[0] not in u_i.keys():
                        u_i[line[0]] = [line[1]]
                    else:
                        u_i[line[0]].append(line[1])
        return u_i

    def save(self):
        file_path = "rec_list.pkl"
        with open(file_path, 'wb') as file:
            pickle.dump(self.Interest_sorting, file)


if __name__ == '__main__':
    dp = data_process('.', "item", 'user', 5, 'data',True)
    dp.save()
