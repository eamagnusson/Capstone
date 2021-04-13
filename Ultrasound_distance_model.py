# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 18:33:32 2021

@author: Brian Coward
"""

## Final Experiment Watch 7
import os
import numpy as np
import pandas as pd
import librosa
from matplotlib import pyplot as plt
from scipy.fft import rfft, fftfreq
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, f1_score


os.chdir("C:\\Users\\Student\\Desktop\\SYS 4055\\FinalExperiment 4-08\\FinalExperiment 4-08\\7\\Audio")

def sqrt(x):
    return x**(1/2)

def mfccFunc(frame, samprate = 44100, n_mfcc = 20): ## Outputs 20 MFCC coefficients for the inputted frame
    mfccs = librosa.feature.mfcc(np.array(frame),samprate, n_mfcc=20)
    y = []
    for e in mfccs:
            y.append(np.mean(e))
        
    return np.array(y)

def fftGraph(frame, samprate = 44100):  ##Returns the x and y vectors of the FFT for the inputted frame.  Easy to be plotted after
    n = len(frame)
    ## Extract FFT for window and associated frequencies
    xf = fftfreq(n, 1/samprate)[0:n//2]  ##Only positive values
    yf = np.abs(rfft(frame)[0:n//2])  
    cutoff = len(xf) * 18000//22050  ##Only show from 18,000 - 20,500 Hz
    end = len(xf) * 20500//22050
    xf = xf[cutoff:end]
    yf = yf[cutoff:end]
    return xf, yf

def fftFunc(frame, samprate = 44100):  ##Outputs FFT amplitudes of frequencies 18,000Hz + 
    n = len(frame)
    yf = np.abs(rfft(frame)[0:n//2])  ## Associated FFT amplitudes for 18,000HZ >
    cutoff = len(yf) * 18000//22050  ##Only show from 18,000 - 22,050 Hz
    yf = yf[cutoff:]
    return yf

def create_data(df, d):  ##Applies FFT and MFCC functions acorss all frames, combines them into one dataset, and inserts the distance column
    mfccs = df.apply(mfccFunc, axis = 0, result_type = 'expand')
    ffts  = df.apply(fftFunc, axis = 0, result_type = 'expand') 
    y = mfccs.append(ffts).transpose()
    y.insert(2 , "Distance", np.repeat(d,119))
    return y

def num_to_cat(x):  ##Create Binary data from Distance for classification
    if x <= 6 :
        return 1
    else:
        return 0
# =============================================================================
# xf = fftfreq(n, 1/samprate)[0:n//2][9000:]  ## 9000 index filters for only 18,000HZ > 
# yf = fft(df.iloc[:,0])[0:n//2][9000:]  ## Associated FFT amplitudes for 18,000HZ >
# y = pd.DataFrame({'xf': xf,'yf':yf}).set_index('xf')
# pd.DataFrame(xf,yf)
# pd.Index(xf)
# =============================================================================

## Load in audio ("ai" variable name means audio at i ft.)
a1, samprate = librosa.load("Swear_audiodata_37c4746a6f972ab5_1617887404334.m4a", sr = None)
a2, samprate = librosa.load("Swear_audiodata_37c4746a6f972ab5_1617887464716.m4a", sr = None)
a3, samprate = librosa.load("Swear_audiodata_37c4746a6f972ab5_1617887525132.m4a", sr = None)
a4, samprate = librosa.load("Swear_audiodata_37c4746a6f972ab5_1617887585481.m4a", sr = None)
a5, samprate = librosa.load("Swear_audiodata_37c4746a6f972ab5_1617887645836.m4a", sr = None)
a6, samprate = librosa.load("Swear_audiodata_37c4746a6f972ab5_1617887706304.m4a", sr = None)
a7, samprate = librosa.load("Swear_audiodata_37c4746a6f972ab5_1617887766742.m4a", sr = None)
a8, samprate = librosa.load("Swear_audiodata_37c4746a6f972ab5_1617887827183.m4a", sr = None)
a9, samprate = librosa.load("Swear_audiodata_37c4746a6f972ab5_1617887887482.m4a", sr = None)
a10, samprate = librosa.load("Swear_audiodata_37c4746a6f972ab5_1617887947755.m4a", sr = None)
a12, samprate = librosa.load("Swear_audiodata_37c4746a6f972ab5_1617888008067.m4a", sr = None)
a15, samprate = librosa.load("Swear_audiodata_37c4746a6f972ab5_1617888068478.m4a", sr = None)
a18, samprate = librosa.load("Swear_audiodata_37c4746a6f972ab5_1617888132856.m4a", sr = None)
a21, samprate = librosa.load("Swear_audiodata_37c4746a6f972ab5_1617888193262.m4a", sr = None)

##  Create rolling time windows from audio vectors

## 1ft  
## starts 16 sec into file
## ends 46 sec

start = 16*samprate  ## Starting indice
end = 46*samprate   ## Ending indice
ft1 = (a1[start:end])
ft1frames = librosa.util.frame(ft1, frame_length=44100//2, hop_length=44100//4, axis=- 1)
df1 = pd.DataFrame(ft1frames)

## 2ft
## starts 15 sec into file
## ends 45 sec

    
start = 15*samprate ## Starting indice
end = 45*samprate ## Ending indice
ft2 = a2[start:end]
ft2frames = librosa.util.frame(ft2, frame_length=44100//2, hop_length=44100//4, axis=- 1)
df2 = pd.DataFrame(ft2frames)


## 3ft
## starts 15 sec into file
## ends 45 secle
    
start = 15* samprate ## Starting indice
end = 45*samprate ## Ending indice
ft3 = a3[start:end]
ft3frames = librosa.util.frame(ft3, frame_length=44100//2, hop_length=44100//4, axis=- 1)
df3 = pd.DataFrame(ft3frames)

## 4ft
## starts 15 sec into file
## ends 45 secle
    
start = 15* samprate ## Starting indice
end = 45*samprate ## Ending indice
ft4 = a4[start:end]
ft4frames = librosa.util.frame(ft4, frame_length=44100//2, hop_length=44100//4, axis=- 1)
df4 = pd.DataFrame(ft4frames)

## 5ft
## starts 14 sec into file
## ends 44 sec
    
start = 14* samprate ## Starting indice
end = 44*samprate ## Ending indice
ft5 = a5[start:end]
ft5frames = librosa.util.frame(ft5, frame_length=44100//2, hop_length=44100//4, axis=- 1)
df5 = pd.DataFrame(ft5frames)

## 6ft
## starts 14 sec into file
## ends 44 secle
    
start = 14* samprate ## Starting indice
end = 44*samprate ## Ending indice
ft6 = a6[start:end]
ft6frames = librosa.util.frame(ft6, frame_length=44100//2, hop_length=44100//4, axis=- 1)
df6 = pd.DataFrame(ft6frames)

## 7ft
## starts 13 sec into file
## ends 43 secle
    
start = 13* samprate ## Starting indice
end = 43*samprate ## Ending indice
ft7 = a7[start:end]
ft7frames = librosa.util.frame(ft7, frame_length=44100//2, hop_length=44100//4, axis=- 1)
df7 = pd.DataFrame(ft7frames)

## 8ft
## starts 13 sec into file
## ends 43 secle
    
start = 13* samprate ## Starting indice
end = 43*samprate ## Ending indice
ft8 = a8[start:end]
ft8frames = librosa.util.frame(ft8, frame_length=44100//2, hop_length=44100//4, axis=- 1)
df8 = pd.DataFrame(ft8frames)

## 9ft
## starts 13 sec into file
## ends 43 secle
    
start = 13* samprate ## Starting indice
end = 43*samprate ## Ending indice
ft9 = a9[start:end]
ft9frames = librosa.util.frame(ft9, frame_length=44100//2, hop_length=44100//4, axis=- 1)
df9 = pd.DataFrame(ft9frames)

## 10ft
## starts 12 sec into file
## ends 42 secle
    
start = 12* samprate ## Starting indice
end = 42*samprate ## Ending indice
ft10 = a10[start:end]
ft10frames = librosa.util.frame(ft10, frame_length=44100//2, hop_length=44100//4, axis=- 1)
df10 = pd.DataFrame(ft10frames)

## 12ft
## starts 12 sec into file
## ends 42 secle
    
start = 12* samprate ## Starting indice
end = 42*samprate ## Ending indice
ft12 = a12[start:end]
ft12frames = librosa.util.frame(ft12, frame_length=44100//2, hop_length=44100//4, axis=- 1)
df12 = pd.DataFrame(ft12frames)

## 15ft
## starts 12 sec into file
## ends 42 secle
    
start = 12* samprate ## Starting indice
end = 42*samprate ## Ending indice
ft15 = a15[start:end]
ft15frames = librosa.util.frame(ft15, frame_length=44100//2, hop_length=44100//4, axis=- 1)
df15 = pd.DataFrame(ft15frames)

## 18ft
## starts 7 sec into file
## ends 37 secle
    
start = 7* samprate ## Starting indice
end = 37*samprate ## Ending indice
ft18 = a18[start:end]
ft18frames = librosa.util.frame(ft18, frame_length=44100//2, hop_length=44100//4, axis=- 1)
df18 = pd.DataFrame(ft18frames)


## 21ft
## starts 15 sec into file
## ends 45 secle
    
start = 7* samprate ## Starting indice
end = 37*samprate ## Ending indice
ft21 = a21[start:end]
ft21frames = librosa.util.frame(ft21, frame_length=44100//2, hop_length=44100//4, axis=- 1)
df21 = pd.DataFrame(ft21frames)


## Now apply functions to create data table
data1 = create_data(df1,1)
data2 = create_data(df2,2)
data3 = create_data(df3,3)
data4 = create_data(df4,4)
data5 = create_data(df5,5)
data6 = create_data(df6,6)
data7 = create_data(df7,7)
data8 = create_data(df8,8)
data9 = create_data(df9, 9)
data10 = create_data(df10,10)
data12 = create_data(df12, 12)
data15 = create_data(df15,15)
data18 = create_data(df18,18)
data21 = create_data(df21, 21)

## create final large dataset
data = pd.concat([data1, data2, data3, data4, data5, data6, data7, data8, data9, data10, data12, data15, data18, data21])

### Now fit model

x = data.drop("Distance", axis=1)
y = data["Distance"]
yCat = y.apply(num_to_cat)

### We’ll use train-test-split to split the data into training data and testing data.
# implementing train-test-split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=66)
x_trainC, x_testC, y_trainC, y_testC = train_test_split(x, yCat, test_size=0.33, random_state=66)

##Now, we can create the random forest model.
##First Regression
rfc = RandomForestRegressor()
rfc.fit(x_train,y_train)
rfc_predict = rfc.predict(x_test)
residuals =np.abs( y_test - rfc_predict)
d = pd.DataFrame({'y':y_test, 'residuals':residuals})
## Total model

avgErr = residuals.mean()
stdErr = np.std(residuals)
sqResid = residuals**2
rmse = np.sqrt(sqResid.mean())

## Within 6 ft

residuals6 = d[d["y"] <= 6].loc[:,"residuals"]
avgErr6 = residuals6.mean()
stdErr6= np.std(residuals6)
sqResid6 = residuals6**2
rmse6 = np.sqrt(sqResid6.mean())

d1 = d.groupby(['y']).mean()

d6 = d.iloc[0:6,:]
sqerrors = d6.loc[:,"residuals"]**2
rmse_6 = np.sqrt(sqerrors.mean())

rfc_cv_score = -1*cross_val_score(rfc, x, y, cv=10, scoring="neg_mean_squared_error")
rmse = np.sqrt(rfc_cv_score)
regMSE = rmse.mean()

## Now Classification
rfcCat = RandomForestClassifier()
rfcCat.fit(x_trainC, y_trainC)
rfcCat_predict = rfcCat.predict(x_testC)

conf = confusion_matrix(y_testC, rfcCat_predict)
tn, fp, fn, tp = conf.ravel()

accuracy = (tn + tp )/ (tn+fp+fn+tp)
f1 = f1_score(y_testC, rfcCat_predict)

rfc_cv_scoreC = cross_val_score(rfcCat, x, yCat, cv = 10, scoring = "accuracy")
rfc_cv_scoreCf1 = cross_val_score(rfcCat, x, yCat, cv = 10, scoring = "f1")

classAccuracy = rfc_cv_scoreC.mean()
classf1 = rfc_cv_scoreCf1.mean()


print("RMSE of Random Forest Regression Model: " + str(round(regMSE,4)))
print("Accuracy of Random Forest Classification Model: " + str(round(classAccuracy,4)))
print("F1 of Random Forest Classification Model: " + str(round(classf1,4)))

x3, y3 = fftGraph(ft3)
x6, y6= fftGraph(ft6)
x9, y9 = fftGraph(ft9)
x12, y12 = fftGraph(ft12)
fig, axs = plt.subplots(2, 2,figsize=(12,12))
axs[0, 0].plot(x3,y3)
axs[0, 0].set_title('3 Feet', fontsize = 20)
axs[0, 1].plot(x6,y6, 'tab:orange')
axs[0, 1].set_title('6 Feet', fontsize = 20)
axs[1, 0].plot(x9,y9, 'tab:green')
axs[1, 0].set_title('9 Feet', fontsize = 20)
axs[1, 1].plot(x12,y12, 'tab:red')
axs[1, 1].set_title('12 Feet', fontsize = 20)

for ax in axs.flat:
    ax.set(xlabel='Frequency', ylabel='Amplitude')
    ax.xaxis.get_label().set_fontsize(15)
    ax.yaxis.get_label().set_fontsize(15)

fig.show()
# Save the full figure...
# fig.savefig('FFTPic.png')


