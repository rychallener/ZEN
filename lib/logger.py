'''
This module contains a little logger class that allows
printing to the screen and to a file. To use:

import logger
import sys
sys.stdout = Logger()
'''

import sys

class Logger():
    def __init__(self, logfile):
        self.terminal = sys.stdout
        self.log      = open(logfile, 'a')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass
