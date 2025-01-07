import random
import json
u_i_history = {}

# 打开第一个文本文件并读取其内容
user = []
item = []
data = []
with open('./dataset/our/data.txt', 'r') as file:
    
    # 逐行读取文件内容
    for line in file:
        # 打印每一行
        line = line.split(" ")
        if len(line) >2:
            if line[0] not in user:
                user.append(line[0])
            if line[1] not in item:
                item.append(line[1])
            line[0] =str(user.index(line[0])) 
            line[1] =str(item.index(line[1])) 
            if u_i_history:
                if line[0] not in u_i_history.keys():
                    u_i_history[line[0]] = [line[1]]
                else:
                    u_i_history[line[0]].append(line[1])
            else:
                u_i_history[line[0]] = [line[1]]

        data.append(line)


with open('output.txt', 'w') as file:
    # 将数组中的每个元素写入文件中
    
    json.dump(u_i_history, file)


train_size = int(len(data) * 0.8)

# 随机选择要分配给训练集的元素
train_set = random.sample(data, train_size)

# 使用列表切片操作获取剩余的元素，作为测试集
test_set = [x for x in data if x not in train_set]

with open('./dataset/our/train_final.txt', 'w') as file:
    # 将数组中的每个元素写入文件中
    for element in train_set:
        if len(element) == 3:
            file.write(element[0]+" "+element[1]+" "+element[2])
with open('./dataset/our/test_final.txt', 'w') as file:
    # 将数组中的每个元素写入文件中
    for element in test_set:
        if len(element) == 3:
            file.write(element[0]+" "+element[1]+" "+element[2])



with open('user_map.txt', 'w') as file:
    # 遍历数组中的每个元素，并将其下标和数值写入文件中
    for index, value in enumerate(user):
        file.write(user[index]+" " +str(index)+"\n")

with open('item_map.txt', 'w') as file:
    # 遍历数组中的每个元素，并将其下标和数值写入文件中
    for index, value in enumerate(item):
        file.write(item[index]+" " +str(index)+"\n")   

with open('user_item_map.txt', 'w') as file:
    # 遍历数组中的每个元素，并将其下标和数值写入文件中
    for index, value in enumerate(user):
        file.write(user[index]+" " +str(index)+"\n")
        

        
            