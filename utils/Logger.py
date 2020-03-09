from datetime import datetime


class Logger:
    def __init__(self):
        self._enabled = True
        self._file = None

    def start(self, filename=""):
        if self._enabled:
            if filename == "":
                self._file = open(str(datetime.timestamp(datetime.now())) + '.log', 'w')
            else:
                self._file = open(filename + '.log', 'w')

    def close(self):
        if self._enabled:
            self._file.close()

    def log(self, msg):
        if self._enabled:
            self._file.write(msg)

    def enable(self):
        self._enabled = True

    def disable(self):
        self._enabled = False
