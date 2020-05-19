#%%

import pandas as pd
from scipy.signal import lfilter
import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial.polynomial import polyfit
import os
from scipy.signal import argrelextrema



Infr2A = pd.read_csv("Infravermelho//2_A.CSV", names=["1", "2"])
Infr6A = pd.read_csv("Infravermelho//6_A.CSV", names=["1", "2"])
Trac2A = pd.read_excel("Tração//Tração_02_A.xlsx")
Trac6C = pd.read_excel("Tração//Tração_06_C.xlsx")


"""

Até agora:
amostra 2 Polietileno (AD, ou BD) ou PP -- Elastômero
amostra 6 PTFE                          -- Elastômero

Ajustar modulo Young (*100) pra tirar a porcentagem

Melhorar amostra 2 ajuste de reta

"""
#%%

df = Infr6A
df["data"] = Infr6A["2"]

lista = []
for x in df["2"]:
    if x<0.1:
        lista.append(0)
    else:
        lista.append(x)

df["data"] = lista


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
# %%
def plot(df, title):
    df2 = df
    df2 = df2.rename(columns={"Position (mm)": "pos","Force (N)": "for", "Strain (%)":"str", "Time (min)": "time"})
    D, L0 = df2["D (mm)"][0], df2["L0 (mm)"][0]
    df2.drop(['Unnamed: 4', 'Unnamed: 5', "D (mm)", "L0 (mm)"], axis=1)


    # plt.plot(df2["pos"], df2["for"])
    # plt.show()
    df2["tension"] = df2["for"]*(1/D)
    df3 = pd.DataFrame(data=[pd.Series(df2["str"]),
                             pd.Series(df2["tension"])]).transpose()
    
    df3 = df3[df3["str"] < 0.799]
    df3 = df3[df3["str"] > 0.1]
    df3 = df3.reset_index()

    
    b, m = polyfit(df3["str"], df3["tension"], 1)
    x = np.linspace(df3["str"][0], df3["str"][len(df3["str"])-1], len(df3["str"]))
    print("Módulo de Young = ", m/10, "(GPa)")

    df2["tension"] = linFilter(df2["tension"])


    plt.plot(df3["str"], df3["tension"])
    plt.plot(x, x*m + b)
    
    plt.title(title)




    plt.xlabel("Strain (%)")
    plt.ylabel("Tension (MPa)")
    
    plt.show()

    plt.plot(df2["pos"]/L0, df2["tension"])
    
    
    plt.title(title)




    plt.xlabel("Pos / L0")
    plt.ylabel("Tension (MPa)")
    limit = sorted([(i,e) for i,e in enumerate(df2["tension"])], key=(lambda x: x[1]), reverse=True)[0]
    print(limit[0])
    a = df2["pos"][limit[0]]/L0
    b = limit[1]
    plt.scatter(a, b, c = "green", label = "Limite de escoamento: %.02f"%b)
    plt.legend()
    
    plt.show()



    # plt.ylim((0,7))
    print()


plot(Trac2A, "Amostra 2")
plot(Trac6C, "Amostra 6")

# %%
def linFilter(y):
    n = 100  # the larger n is, the smoother curve will be
    b = [1.0 / n] * n
    a = 1
    yy = lfilter(b,a,y)
    return yy

def findLocalMaximum(strain, start, end, force):
    return max(force[np.logical_and(strain>start,strain<end)])

def chopTail(arr, val):
    return np.where(arr<val)
# %%
