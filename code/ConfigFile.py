import configparser as ConfigParser

global config

strFilePath= "data.properties"
def getProperty(propName):

    return config.get("deepbind",propName)


def reloadProperties():
    config.read(strFilePath)

print("Loading property file")
config = ConfigParser.RawConfigParser()
config.read(strFilePath)


if __name__ == '__main__':

    print (config.get("deepbind","implicit.data.file"))
