import pandas as pd
import numpy as np
from keras.models import load_model
import mVAE_helper
import ConfigFile

# model = load_model("../model/gan_gen.pkl")

model = load_model("/Users/raghuramsrinivas/localdrive/education/deepbind/paper2/model/gan_gen.pkl")
noise = np.random.normal(0, 1, (5, 292))

out = model.predict(noise)

# print(out)

"""
for i in range(0, out.shape[0]):
                                                    
	retStr = mVAE_helper.isValidEncoding(out[i,:])
	print(retStr)

print("Done.")
"""

featuresFile = pd.read_csv(ConfigFile.getProperty("implicit.data.file"))

encodedColNames = ["%d_latfeatures" % i for i in range(0,
                                                       int(ConfigFile.getProperty("encoded.feature.size")))]

for i in range(0, 40):
    tempArr = featuresFile.loc[i, encodedColNames]
    print(mVAE_helper.isValidEncoding(tempArr))
print (tempArr.shape)

featuresFile.loc[40, encodedColNames]
mVAE_helper.isValidEncoding(tempArr)
