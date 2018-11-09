#import configparser as ConfigParser

import  ConfigParser
global config

strFilePath= "/Users/raghuramsrinivas/localdrive/education/deepbind/paper2/src/data.properties"
#strFilePath=

def getProperty(propName):

    return config.get("deepbind",propName)


def reloadProperties():
    config.read(strFilePath)

print("Loading property file")
config = ConfigParser.RawConfigParser()
config.read(strFilePath)


if __name__ == '__main__':

    print (config.get("deepbind","implicit.data.file"))
    print(getProperty("encoded.feature.size"))
