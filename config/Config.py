import json
import os


class Config(object):
    def __init__(self, data, name=None):
        self.name = name
        for key in data:
            value = data[key]
            key = key.strip()
            if type(value) is dict:
                self.__dict__[key] = Config(value)
            else:
                self.__dict__[key] = value

    def check(self, key):
        result = False
        if hasattr(self, key):
            result = self.__dict__[key]
        return result

    def get(self, key):
        result = None
        for k in self.__dict__:
            if self.__dict__[k] is Config:
                result = self.__dict__[k].get(key)
            else:
                if k == key:
                    result = self.__dict__[k]

        return result

    @staticmethod
    def parse_config(filename):
        with open(filename + '.json') as config_file:
            data = json.load(config_file)

        return Config(data, name=os.path.split(filename)[1])
