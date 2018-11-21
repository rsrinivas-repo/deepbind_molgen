
import  os

envInfo = os.getenv("deepbind_env")
if envInfo is None:
    print("Set environment variable deepbind_env as local or m2 ")
    exit()

if envInfo.strip()=="local":
    import  ConfigParser
    strFilePath = "/Users/raghuramsrinivas/localdrive/education/deepbind/paper2/src/data.properties"
    configBlock="deepbind_local"

else:
    import configparser as ConfigParser
    strFilePath="/users/rsrinivas/deepbind/paper2/deepbind_molgen/src/data.properties"
    configBlock="deepbind_m2"


global config

def getProperty(propName):

    return config.get(configBlock,propName)


def reloadProperties():
    config.read(strFilePath)

print("Loading property file")
config = ConfigParser.RawConfigParser()
config.read(strFilePath)


if __name__ == '__main__':

    print (config.get(configBlock,"implicit.data.file"))
    print(getProperty("encoded.feature.size"))
