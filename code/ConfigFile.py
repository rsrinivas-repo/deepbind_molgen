import ConfigParser

global config
def getProperty(propName):

    return config.get("deepbind",propName)


def reloadProperties():
    config.read('/Users/raghuramsrinivas/localdrive/education/deepbind/paper2/code/data.properties')

print("Loading property file")
config = ConfigParser.RawConfigParser()
config.read('/Users/raghuramsrinivas/localdrive/education/deepbind/paper2/code/data.properties')


if __name__ == '__main__':

    print (config.get("deepbind","implicit.data.file"))