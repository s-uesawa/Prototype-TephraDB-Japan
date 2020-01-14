# -*- coding: utf-8 -*-
"""
Created on Mon Feb 18 18:11:05 2019

@author: Shimpei Uesawa
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.special import factorial
from scipy import stats

#F = "Tokyo"   
### If you cannot connect with R to Python, please input the name of the place where you want to draw the hazard curve and delete the '#' before 'F'. Run directly with Pyhon.

df1 = pd.read_csv('C:/Users/your_dir/TephraDB_Prototype_ver1/combinedPointValue.csv') 
#### Please enter your directory. Don't change the last file name.

df2 = pd.read_csv('C:/Users/Your_dir/TephraDB_Prototype_ver1/No_and_age_list_fin.csv')
#### Please enter your directory. Don't change the last file name.
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
df.to_csv('C:/Users//Your_dir/TephraDB_Prototype_ver1/Tephra_Fall_History.csv')
#### Please enter your directory. Don't change the last file name.

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
df_boolTot = df.query('(Year_ka<150)')

w3 = np.linspace(1,2800,280)
w2 = np.array([0,0.1])
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

Z = []
Zav = []
Parameters = []
Prob1 = []
Prob2 = []
Prob3 = []
Conf_95 = []
mean = []
Pc = []

n = len(w)
n2 = range(0,n)

def poisson(k, lamb):
    return (lamb**k/factorial(k)*np.exp(-lamb))

#calculation of 95% confidence interval

alpha = 0.95
data = []
for i in n2:
    data1 = J1np[i] # recent 1 ka
    Zav1 = np.average(data1)/10000
    data2 = JTot1np[:,i] # recent 2 ka
    Zav2 = np.average(data2)/10000
    data3 = JTot2np[:,i] # recent 3 ka
    Zav3 = np.average(data3)/10000
    data4 = JTot3np[:,i] # recent 4 ka
    Zav4 = np.average(data4)/10000
    data5 = JTot4np[:,i] # recent 5 ka
    Zav5 = np.average(data5)/10000
    data6 = JTot5np[:,i] # recent 6 ka
    Zav6 = np.average(data6)/10000
    data7 = JTot6np[:,i] # recent 7 ka
    Zav7 = np.average(data7)/10000
    data8 = JTot7np[:,i] # recent 8 ka
    Zav8 = np.average(data8)/10000
    data9 = JTot8np[:,i] # recent 9 ka
    Zav9 = np.average(data9)/10000
    data10 = JTot9np[:,i] # recent 10 ka
    Zav10 = np.average(data10)/10000
    data11 = JTot10np[:,i] # recent 11 ka
    Zav11 = np.average(data11)/10000
    data12 = JTot11np[:,i] # recent 12 ka
    Zav12 = np.average(data12)/10000
    data13 = JTot12np[:,i] # recent 13 ka
    Zav13 = np.average(data13)/10000
    data14 = JTot13np[:,i] # recent 14 ka
    Zav14 = np.average(data14)/10000
    data15 = JTot14np[:,i] # recent 15 ka
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

def poisson2(t, lamb2):
    return 1 - np.exp(-lamb2*t)

for i in mean:
    Pc1 = poisson(0, i)
    Pc.append(Pc1)

for i in mean:  #Zav or Parameters calculation of annual exceeding probability
    a = poisson2(1, i)
    Prob1.append(a)

for i in n2:  #Zav or Parameters calculation of annual exceeding probability
    max_95_1 = Conf_95[i]
    max_95_2 = max_95_1[1]
    a = poisson2(1, max_95_2)
    Prob2.append(a)    

for i in n2:  #Zav or Parameters calculation of annual exceeding probability
    min_95_1 = Conf_95[i]
    min_95_2 = min_95_1[0]
    a = poisson2(1, min_95_2)
    Prob3.append(a) 

#Exponential decay
def func1(x1, a1, b1):
    return np.exp(-b1*x1)*a1

popt1, pcov1 = curve_fit(func1, w, Prob1)
popt1

#Weibull proposed by Tatsumi and Suzuki
def func2(x2, a2, b2, c2):
    return np.exp(-(x2/a2)**b2)*c2

popt2, pcov2 = curve_fit(func2, w, Prob1)
popt2

#R^2 for func1
r1 = Prob1 - func1(w, *popt1)
rr1 = Prob1 - np.average(Prob1)
R21 = 1 -(sum((r1)**2)/(sum((rr1)**2)))


#R^2 for func2
r2 = Prob1 - func2(w, *popt2)
rr2 = Prob1 - np.average(Prob1)
R22 = 1 -(sum((r2)**2)/(sum((rr2)**2)))


#likelihood function for func1
numpara1 = 1
 
def func4(x4, lamda4):
    return np.log(lamda4)*len(x4) - lamda4*sum(x4)

x31 = []

for i in w:
    if func1(i, *popt1) > 10**(-6):
        x31.append(i)
    else:
        print('end')

#Muximum likelyhood lamda for func1

AIC1 = (-2)*func4(x31, popt1[1]) + 2*numpara1
AICc1 = AIC1 + ((2*numpara1*(numpara1+1))/(len(w)-numpara1- 1))

#likelihood function for func2
numpara2 = 2

x32 = []
LE = []
H = []
I = []

for i in w:
    if func2(i, *popt2) > 10**(-6):
        x32.append(i)
    else:
        print('end')

N = len(x32)

for i in x32:
    G = np.log(i)
    H.append(G)

for i in x32:
    K = (i/popt2[0])**popt2[1]    
    I.append(K)

H2 = H[1:]

#Muximum likelyhood lamda for func2
MLE = N*np.log(popt2[1]) - popt2[1]*N*np.log(popt2[0]) + (popt2[1]-1)*sum(H2) - sum(I)

AIC2 = (-2)*MLE + 2*numpara2
AICc2 = AIC2 + ((2*numpara2*(numpara2 + 1))/(len(x32) - numpara2 - 1))

print('Constants and coefficents:')
print('Exp. Decay, [a1, b1]=', popt1) # display in the order of [a1, b1]
print('Weibull, [b2, c2, a2]=', popt2) # display in the order of [b2, c2, a2]

print('R^2=')
print('Exp. Decay =', R21)
print('Weibull =', R22)

print('AIC=')
print('Exp. Decay', AIC1)
print('Weibull', AIC2)

print('AICc=')
print('Exp. Decay', AICc1)
print('Weibull', AICc2)

w2 = np.linspace(1,2801,281)

plt.figure(figsize=(5,5))
plt.loglog(w, Prob1, color='black')
plt.loglog(w, Prob2, '--', color='black', linewidth=0.5)
plt.loglog(w, Prob3, '--', color='black', linewidth=0.5)
plt.ylim(0.000001, 0.001)
plt.xlim(0,3000)
plt.title(F)
plt.xlabel(r'Tephra fall load $\mathrm{(kg/m^2)}$')
plt.ylabel('Annual exceedance probability')
plt.loglog(w, func1(w, *popt1), 'r--', label='Exponential')
plt.loglog(w, func2(w, *popt2), 'b--', label='Weibull')
plt.legend()
plt.savefig(F, dpi=150)
plt.show()
