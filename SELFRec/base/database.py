import pickle
from sshtunnel import SSHTunnelForwarder
import pymysql
import sys
import os
import json
import pandas as pd
from tqdm import tqdm
class database:
    def __init__(self, host, user,
                 password, port,
                 database,
                 sshset=None,
                 arg=None):
        self.host = host
        self.user = user
        self.port = int(port)
        self.password = password
        self.database = database
        if sshset!=None:
            self.cnx = self.proxy_connect(sshset)
        # 建立数据库连接
        else:
            self.cnx = self.connect()

        # 创建游标对象
        self.cursor = self.cnx.cursor()

        # 使用json.loads()将字符串转换为字典
        if arg != None:
            self.table = json.loads(arg['table'].replace("'", "\""))
            for table in self.table:
                self.existing_table(table)

    def connect(self):
        # 判断数据库是否存在
        connection = pymysql.connect(
            host=self.host,
            port=self.port,
            user=self.user,
            password=self.password
        )
        # 数据库名称
        db_name = self.database

        # 使用 cursor() 方法创建一个游标对象
        cursor = connection.cursor()

        # 查询数据库是否存在
        cursor.execute(f"SHOW DATABASES LIKE '{db_name}'")

        # 检查是否有结果
        database_exists = cursor.fetchone() is not None

        if not database_exists:
            # 如果数据库不存在，创建数据库
            cursor = connection.cursor()
            cursor.execute(f"CREATE DATABASE {db_name}")
            cursor.close()
        else:
            cursor.close()

        return pymysql.connect(
            host=self.host,
            user=self.user,
            password=self.password,
            database=self.database,
            port=self.port
        )

    def proxy_connect(self, sshset):
        sshset = eval(sshset)
        ssh_config = {
            'ssh_address': (sshset['host'], sshset['port']),
            'ssh_password': sshset['password'],
            'ssh_username': sshset['username'],
            'remote_bind_address': (self.host,self.port)
        }
        server = SSHTunnelForwarder(**ssh_config)
        server.start()

        # 判断数据库是否存在
        connection = pymysql.connect(
            host='127.0.0.1',
            port=server.local_bind_port,
            user=self.user,
            password=self.password
        )
        # 数据库名称
        db_name = self.database

        # 使用 cursor() 方法创建一个游标对象
        cursor = connection.cursor()

        # 查询数据库是否存在
        cursor.execute(f"SHOW DATABASES LIKE '{db_name}'")

        # 检查是否有结果
        database_exists = cursor.fetchone() is not None

        if not database_exists:
            # 如果数据库不存在，创建数据库
            cursor = connection.cursor()
            cursor.execute(f"CREATE DATABASE {db_name}")
            cursor.close()
        else:
            cursor.close()


        database = pymysql.connect(
            host='127.0.0.1',
            port=server.local_bind_port,
            user=self.user,
            password=self.password,
            database=self.database
        )
        return database
    
    # 表格是否存在 不存在则创建
    def existing_table(self,table):
        # 表名称
        table_name = table

        # 使用 cursor() 方法创建一个游标对象
        cursor = self.cursor

        # 查询表是否存在
        cursor.execute(f"SHOW TABLES LIKE '{table_name}'")

        # 检查是否有结果
        table_exists = cursor.fetchone() is not None
        print()
        if not table_exists:
            # 如果表不存在，创建表
          
            # 编写创建表的SQL语句，例如：
            create_table_sql = """
            CREATE TABLE {table_name} (
                {column1} int,
                {column2} tinytext
            )
            """.format(table_name=table_name,column1=self.table[table][0],column2=self.table[table][1])
            cursor.execute(create_table_sql)
            
    # def readTable_pd(self,table):
    #
    #     query = "SELECT * FROM " + table
    #     return pd.read_sql_query(query, self.cnx)

    def readTable_pd(self, table):
        query = "SELECT * FROM " + table
        print("SELECT COUNT(*) FROM ({query}) AS subquery", self.cnx)
        # 获取查询结果的总行数
        total_rows = pd.read_sql_query(f"SELECT COUNT(*) FROM ({query}) AS subquery", self.cnx)['COUNT(*)'].values[0]

        # 创建进度条
        with tqdm(total=total_rows, desc="Reading Data") as pbar:

            # 逐块读取数据并更新进度条
            chunk_size = 100  # 你可以根据需要自定义块大小
            result = None  # 用于存储结果的 DataFrame
            for chunk in pd.read_sql_query(query, self.cnx, chunksize=chunk_size):
                # 在这里处理每个数据块
                # 例如，你可以将数据块追加到结果 DataFrame 中
                if result is None:
                    result = chunk
                else:
                    result = pd.concat([result, chunk], ignore_index=True)

                # 更新进度条
                pbar.update(chunk_size)

        return result

    def writeTable(self, table, title, value):
        # # 执行 SQL 查询语句
        # query = "INSERT INTO " + table + " "
        # query += "(" + ",".join(title) + ")"
        # query += " " + "VALUES (%s, %s)"
        # # print(query)
        # self.cursor.execute(query, value)
        # self.cnx.commit()
        # 执行 SQL 插入语句
        print('%s'%table)
        query = "INSERT INTO " + table + " "
        query += "(" + ",".join(title) + ")"
        query += " VALUES (" + ",".join(["%s"] * len(title)) + ")"
        print('%s'%query)
        self.cursor.executemany(query, [value])
        self.cnx.commit()

    def deleteTable(self, table):
        query = "DELETE FROM " + table
        self.cursor.execute(query)
        self.cnx.commit()
        # 自增
        query = "ALTER TABLE " + table + ' AUTO_INCREMENT = 1'
        self.cursor.execute(query)
        self.cnx.commit()

    def disconnect(self):
        self.cursor.close()
        self.cnx.close()

    def load_rec_list(self,path='.'):
        file_path = os.path.join(path,"rec_list.pkl")
        with open(file_path, 'rb') as file:
            loaded_data = pickle.load(file)
        return loaded_data


