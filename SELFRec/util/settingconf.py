import os
import json
import configparser

class SettingConf(object):
    def __init__(self,file):
        self.conf = configparser.ConfigParser()
        self.conf.read(file,encoding='utf-8')
        self.config = self.read_configuration()

    def read_configuration(self):
        data = {}
        for section in self.conf.sections():
            conf_data = {}
            for key, value in self.conf.items(section):
                conf_data[key] = value
            data[section] = conf_data
        return data