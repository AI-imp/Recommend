import numpy as np
from scipy.io import savemat
from scipy import io
smiuser_num = 3  # 用户数量
cell_array = np.empty((smiuser_num,), dtype=np.object)
for i in range(smiuser_num):
    cell_array[i] = np.array([100283363,100283436,100283971])  # 数组存放每个用户的pos数据，当前登录用户放在第一位为0
cell_matrix = np.empty((smiuser_num, 1), dtype=np.object)
cell_matrix[:, 0] = cell_array
a = {'neg_data': cell_matrix}
savemat(r'F:\python\learningproject\nlprecommend\Deep_Matrix_Factorization_Models-master-master\Deep_Matrix_Factorization_Models-master\Data\neg_data.mat', a)
