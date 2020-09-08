import json
import os


class Config(object):
    def __init__(self, data, name=None):
        self.name = name
        for key in data:
            value = data[key]
            if type(value) is dict:
                self.__dict__[key] = Config(value)
            else:
                self.__dict__[key] = value

    @staticmethod
    def parse_config(filename):
        with open(filename + '.json') as config_file:
            data = json.load(config_file)

        return Config(data, name=os.path.split(filename)[1])
