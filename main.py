#%%

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema



Infr2A = pd.read_csv("Infravermelho//2_A.CSV", names=["1", "2"])
Infr6A = pd.read_csv("Infravermelho//6_A.CSV", names=["1", "2"])
Trac2A = pd.read_excel("Tração//Tração_02_A.xlsx")
Trac6C = pd.read_excel("Tração//Tração_06_C.xlsx")


#%%

df = Infr2A
df["data"] = Infr2A["2"]

lista = []
for x in df["2"]:
    if x<0.05:
        lista.append(0)
    else:
        lista.append(x)

df["2"] = lista


n=15 # number of points to be checked before and after 
# Find local peaks
df['min'] = df.iloc[argrelextrema(df.data.values, np.less, order=n)[0]]['data']
df['max'] = df.iloc[argrelextrema(df.data.values, np.greater, order=n)[0]]['data']

# Plot results
# plt.figure(figsize=(21,12))
plt.scatter(df["1"], df['min'], c='r')
plt.scatter(df["1"], df['max'], c='g')
plt.plot(df["1"], df['data'])
plt.show()

print(df["1"][df["max"].dropna().index])


df

# %%
