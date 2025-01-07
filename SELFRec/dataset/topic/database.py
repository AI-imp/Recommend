
import pickle
from sshtunnel import SSHTunnelForwarder
import pymysql

class database:
    def __init__(self, host='10.240.97.164', user='zzj',
                 password='@zzj121', port=3306,
                 database='T_newsRecommended',
                 use_proxy=False,
                 sshset=None):
        self.host = host
        self.user = user
        self.port = port
        self.password = password
        self.database = database
        if use_proxy:
            self.cnx = self.proxy_connect(sshset)
        # 建立数据库连接
        else:
            self.cnx = self.connect()

        # 创建游标对象
        self.cursor = self.cnx.cursor()

    def connect(self):
        return pymysql.connect(
            host=self.host,
            user=self.user,
            password=self.password,
            database=self.database,
            port=self.port
        )

    def proxy_connect(self, sshset):
        """
                sshset = {
                "host": "jie88.top",
                "port": 6000,
                "username": 'zzj',
                "password": "Zzj@123."
            }
        """
        ssh_config = {
            'ssh_address': (sshset['host'], sshset['port']),
            'ssh_password': sshset['password'],
            'ssh_username': sshset['username'],
            'remote_bind_address': ('10.240.97.164', 3306)
        }
        server = SSHTunnelForwarder(**ssh_config)
        server.start()

        database = pymysql.connect(
            host='127.0.0.1',
            port=server.local_bind_port,
            user=self.user,
            password=self.password,
            database=self.database
        )
        return database

    def readTable(self, table):
        # 执行 SQL 查询语句
        query = "SELECT * FROM " + table
        self.cursor.execute(query)
        # 获取查询结果
        result = self.cursor.fetchall()
        return result

    def writeTable(self, table, tittle, value):
        # 执行 SQL 查询语句
        query = "INSERT INTO " + table + " "
        query += "(" + ",".join(tittle) + ")"
        query += " " + "VALUES (%s, %s)"
        # print(query)
        self.cursor.execute(query, value)
        self.cnx.commit()

    def deleteTable(self, table):
        query = "DELETE FROM " + table
        self.cursor.execute(query)
        self.cnx.commit()

    def disconnect(self):
        self.cursor.close()
        self.cnx.close()

    def load_rec_list(self):
        file_path = "rec_list.pkl"
        with open(file_path, 'rb') as file:
            loaded_data = pickle.load(file)
        return loaded_data


if __name__ == '__main__':
    table = "news_to_user"
    # db = database(use_proxy=True, sshset=sshset)
    db = database()
    # data = db.load_rec_list()
    # # 清空后写入
    # db.deleteTable(table)
    # for user_id, news_id in data.items():
    #     db.writeTable(table, ["news_id", "user_id"], (user_id, str(news_id)))

    data = db.readTable(table)
    print(data)
    db.disconnect()
