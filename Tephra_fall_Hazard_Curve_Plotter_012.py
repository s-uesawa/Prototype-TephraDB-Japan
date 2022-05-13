# -*- coding: utf-8 -*-
"""
Created on Mon Feb 18 18:11:05 2019

@author: Shimpei Uesawa
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import factorial
from scipy import stats

df1 = pd.read_csv('C:/TephraDB_Prototype_ver1.1/combinedPointValue_012.csv') 
#### Please enter your directory. Don't change the foler and the last file name.

df2 = pd.read_csv('C:/TephraDB_Prototype_ver1.1/No_and_age_list_fin.csv')
#### Please enter your directory. Don't change the foler and the last file name.

df3 = df1.fillna(0)
df4 = df3.T

df4.columns = df4.iloc[0]
df5 = df4.drop(labels='Name')

df6 = df5.reset_index()
df6['index'] = pd.to_numeric(df6['index'].str.replace('No_', ''))

df7 = df2.sort_values("No.")
df8 = df6.sort_values("index")
df9 = df8.set_index("index")
df10 = df7.set_index("No.")

df = pd.merge(df9, df10, how='outer', left_index=True, right_index=True)

df.to_csv('C:/TephraDB_Prototype_ver1.1/Tephra_Fall_History_012.csv') ## Please input your directory

### If you run the R script, you can skip lines 14 through 36 and use the following script, which removes the "#" in front of the df in line 39. ###
#df = pd.read_csv('C:/Users/Your_directory/TephraDB_Prototype_ver1.1/Tephra_Fall_History.csv') ## Please input your directory

F = "Tokyo"

df_bool1 = df.query('(Year_ka >=0)&(Year_ka <10)')
df_bool2 = df.query('(Year_ka >=10)&(Year_ka <20)')
df_bool3 = df.query('(Year_ka >=20)&(Year_ka <30)')
df_bool4 = df.query('(Year_ka >=30)&(Year_ka<40)')
df_bool5 = df.query('(Year_ka >=40)&(Year_ka<50)')
df_bool6 = df.query('(Year_ka >=50)&(Year_ka<60)')
df_bool7 = df.query('(Year_ka >=60)&(Year_ka<70)')
df_bool8 = df.query('(Year_ka >=70)&(Year_ka <80)')
df_bool9 = df.query('(Year_ka >=80)&(Year_ka <90)')
df_bool10 = df.query('(Year_ka >=90)&(Year_ka <100)')
df_bool11 = df.query('(Year_ka >=100)&(Year_ka<110)')
df_bool12 = df.query('(Year_ka >=110)&(Year_ka <120)')
df_bool13 = df.query('(Year_ka >=120)&(Year_ka <130)')
df_bool14 = df.query('(Year_ka >=130)&(Year_ka <140)')
df_bool15 = df.query('(Year_ka >=140)&(Year_ka<150)')
df_boolTot = df.query('(Year_ka<=150)')

w3 = np.linspace(10,3000,300)
w2 = np.array([0,0.1,1])
w = np.append(w2,w3)

J1 = []
J2 = []
J3 = []
J4 = []
J5 = []
J6 = []
J7 = []
J8 = []
J9 = []
J10 = []
J11 = []
J12 = []
J13 = []
J14 = []
J15 = []
JTot = []

for i in w:
    Fi = (df_bool1[F] > i)
    Fin = Fi.sum()
    J1.append(Fin)

for i in w:
    Fi = (df_bool2[F] > i)
    Fin = Fi.sum()
    J2.append(Fin)
    
for i in w:
    Fi = (df_bool3[F] > i)
    Fin = Fi.sum()
    J3.append(Fin)
    
for i in w:
    Fi = (df_bool4[F] > i)
    Fin = Fi.sum()
    J4.append(Fin)
    
for i in w:
    Fi = (df_bool5[F] > i)
    Fin = Fi.sum()
    J5.append(Fin)
    
for i in w:
    Fi = (df_bool6[F] > i)
    Fin = Fi.sum()
    J6.append(Fin)
    
for i in w:
    Fi = (df_bool7[F] > i)
    Fin = Fi.sum()
    J7.append(Fin)
    
for i in w:
    Fi = (df_bool8[F] > i)
    Fin = Fi.sum()
    J8.append(Fin)
    
for i in w:
    Fi = (df_bool9[F] > i)
    Fin = Fi.sum()
    J9.append(Fin)
    
for i in w:
    Fi = (df_bool10[F] > i)
    Fin = Fi.sum()
    J10.append(Fin)
    
for i in w:
    Fi = (df_bool11[F] > i)
    Fin = Fi.sum()
    J11.append(Fin)

for i in w:
    Fi = (df_bool12[F] > i)
    Fin = Fi.sum()
    J12.append(Fin)
    
for i in w:
    Fi = (df_bool13[F] > i)
    Fin = Fi.sum()
    J13.append(Fin)
    
for i in w:
    Fi = (df_bool14[F] > i)
    Fin = Fi.sum()
    J14.append(Fin)
    
for i in w:
    Fi = (df_bool15[F] > i)
    Fin = Fi.sum()
    J15.append(Fin)
    
for i in w:
    Fi = (df_boolTot[F] > i)
    Fin = Fi.sum()
    JTot.append(Fin)

y = np.array(JTot)/150000

J1np = np.array(J1)
J2np = np.array(J2)
J3np = np.array(J3)
J4np = np.array(J4)
J5np = np.array(J5)
J6np = np.array(J6)
J7np = np.array(J7)
J8np = np.array(J8)
J9np = np.array(J9)
J10np = np.array(J10)
J11np = np.array(J11)
J12np = np.array(J12)
J13np = np.array(J13)
J14np = np.array(J14)
J15np = np.array(J15)

JTot1np = np.vstack((J1np,J2np))
JTot2np = np.vstack((JTot1np,J3np))
JTot3np = np.vstack((JTot2np,J4np))
JTot4np = np.vstack((JTot3np,J5np))
JTot5np = np.vstack((JTot4np,J6np))
JTot6np = np.vstack((JTot5np,J7np))
JTot7np = np.vstack((JTot6np,J8np))
JTot8np = np.vstack((JTot7np,J9np))
JTot9np = np.vstack((JTot8np,J10np))
JTot10np = np.vstack((JTot9np,J11np))
JTot11np = np.vstack((JTot10np,J12np))
JTot12np = np.vstack((JTot11np,J13np))
JTot13np = np.vstack((JTot12np,J14np))
JTot14np = np.vstack((JTot13np,J15np))

Cum_freq_150 = []

n = np.linspace(0,302,303, dtype=int)

for i in n:
    J = JTot14np[:,i]
    Cum = sum(J)
    Cum_freq_150.append(Cum)
    
Z = []
Zav = []
Parameters = []
Prob1 = []
Conf_95_max = []
Conf_95_min = []
Conf_95 = []
mean = []
Pc = []

n = len(w)
n2 = range(0,n)

# for CDF
def poisson(k, lamb):
    return (lamb**k/factorial(k)*np.exp(-lamb))

#calculation of 95% confidence interval

alpha = 0.95
data = []
for i in n2:
    data1 = J1np[i] # recent 10 ka
    Zav1 = np.average(data1)/10000
    data2 = JTot1np[:,i] # recent 20 ka
    Zav2 = np.average(data2)/10000
    data3 = JTot2np[:,i] # recent 30 ka
    Zav3 = np.average(data3)/10000
    data4 = JTot3np[:,i] # recent 40 ka
    Zav4 = np.average(data4)/10000
    data5 = JTot4np[:,i] # recent 50 ka
    Zav5 = np.average(data5)/10000
    data6 = JTot5np[:,i] # recent 60 ka
    Zav6 = np.average(data6)/10000
    data7 = JTot6np[:,i] # recent 70 ka
    Zav7 = np.average(data7)/10000
    data8 = JTot7np[:,i] # recent 80 ka
    Zav8 = np.average(data8)/10000
    data9 = JTot8np[:,i] # recent 90 ka
    Zav9 = np.average(data9)/10000
    data10 = JTot9np[:,i] # recent 100 ka
    Zav10 = np.average(data10)/10000
    data11 = JTot10np[:,i] # recent 110 ka
    Zav11 = np.average(data11)/10000
    data12 = JTot11np[:,i] # recent 120 ka
    Zav12 = np.average(data12)/10000
    data13 = JTot12np[:,i] # recent 130 ka
    Zav13 = np.average(data13)/10000
    data14 = JTot13np[:,i] # recent 140 ka
    Zav14 = np.average(data14)/10000
    data15 = JTot14np[:,i] # recent 150 ka
    Zav15 = np.average(data15)/10000
    Zav16 = np.vstack((Zav1,Zav2))
    Zav17 = np.vstack((Zav16,Zav3))
    Zav18 = np.vstack((Zav17,Zav4))
    Zav19 = np.vstack((Zav18,Zav5))
    Zav20 = np.vstack((Zav19,Zav6))
    Zav21 = np.vstack((Zav20,Zav7))
    Zav22 = np.vstack((Zav21,Zav8))
    Zav23 = np.vstack((Zav22,Zav9))
    Zav24 = np.vstack((Zav23,Zav10))
    Zav25 = np.vstack((Zav24,Zav11))
    Zav26 = np.vstack((Zav25,Zav12))
    Zav27 = np.vstack((Zav26,Zav13))
    Zav28 = np.vstack((Zav27,Zav14))
    Zav29 = np.vstack((Zav28,Zav15)) 
    mean_val = np.mean(Zav29)
    mean.append(mean_val)
    sem_val = stats.sem(Zav29)
    ci = stats.t.interval(alpha,  len(Zav29)- 1, loc=mean_val, scale=sem_val)
    Conf_95.append(ci)

for i in n2:  #Zav or Parameters calculation of annual exceeding probability
    max_95_1 = Conf_95[i]
    max_95_2 = max_95_1[1]
    Conf_95_max.append(max_95_2)    

for i in n2:  #Zav or Parameters calculation of annual exceeding probability
    min_95_1 = Conf_95[i]
    min_95_2 = min_95_1[0]
    Conf_95_min.append(min_95_2) 
    
dfCum = pd.DataFrame(Cum_freq_150)

w2 = np.linspace(1,2801,281)

fig = plt.figure(figsize=(10,5))
ax2, ax= fig.subplots(nrows=1, ncols=2)

ax.loglog(w, mean, color='black')
ax.loglog(w, Conf_95_max, '--', color='red', linewidth=0.5)
ax.loglog(w, Conf_95_min, '--', color='red', linewidth=0.5)
ax.set_ylim(0.000001, 0.001)
ax.set_xlim(0.1,3000)
ax.set_title(F)
##ax.set_xlabel(r'Tephra fall load $\mathrm{(kg/m^2)}$')
ax.set_xlabel('Tephra fall thickness (mm)')
ax.set_ylabel('Mean annual frequency of exceedance')

ax2.semilogx(w, dfCum[0], color='black')
ax2.set_ylim(0, 25)
ax2.set_xlim(0.1,3000)
ax2.set_title(F)
ax2.set_xlabel('Tephra fall thickness (mm)')
ax2.set_ylabel('Number of exceeding tephra fall thickness')

fig.show()
