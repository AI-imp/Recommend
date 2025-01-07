import pandas as pd
import numpy as np
from scipy import io
#活跃用户和等待推荐物品还要自己定义
#第一次转
# user_to_item = {}
# pos_data = io.loadmat(
#             r'F:\python\learningproject\nlprecommend\Deep_Matrix_Factorization_Models-master-master\Deep_Matrix_Factorization_Models-master\Data\pos_data.mat')  # 读取active_id
# for index, i in enumerate(pos_data['pos_data'], start=1):
#     user_to_item[index] = i[0][0].tolist()
# print(user_to_item)
# data_available_item = io.loadmat(
#             r'F:\python\learningproject\nlprecommend\Deep_Matrix_Factorization_Models-master-master\Deep_Matrix_Factorization_Models-master\Data\n_sim.mat')  # 读取available_item_id
# data_available = data_available_item['n_sim'].astype(int)[:, 0]
# item_map = {}  # 同上
# for index, item_id in enumerate(data_available, start=1):
#     item_map[item_id] = index
# print(item_map)
a=io.loadmat(r'F:\python\learningproject\nlprecommend\Deep_Matrix_Factorization_Models-master-master\Deep_Matrix_Factorization_Models-master\Data\recommend_item.mat')
a=a['recommend_item']
print(a[1])





