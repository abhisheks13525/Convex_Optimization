#EE20RESCH11005_LAB_7
import cvxpy as cp
import numpy as np
from numpy import savetxt
from numpy import loadtxt
import math
import matplotlib.pyplot as plt
c = loadtxt('data.npy')
p=np.reshape(p,[2016,1])

N=288
T =2016 

# Defining Variables

c=cp.Variable((T,1)) # output for clear sky
s=cp.Variable((T,1)) # shading loss component with cloud
r=cp.Variable((T,1)) # residual errors

### Defining Objective Function for the problem

loss_f=sum((c[i]-c[i+1])**2 for i in range(1,N))+(c[N-1]-c[0])**2
loss_f+=np.ones(T).T@s

#Defining Objective  
objective=cp.Minimize(loss_f)


### Defining  Constraints for the defined objective

constraints=[0<=s,s<=c,p==c-s+r]
constraints+=[(cp.norm(r,1)/T)<=4]
constraints+=[(c[i]==c[i-N]) for i in range(N,T)]

# Formulating convex problem

prob=cp.Problem(objective,constraints)
prob.solve()



#Printing required results 
print('Minimum value of loss function is given as :',prob.solve(),"\n\n")       
#print(c.value)
print(' The avg value of  output clear sky (c) :',np.mean(c.value),"\n\n")
print('The avg value of shading loss component (s):',np.mean(s.value ),"\n \n")
print(' The avg value of Photo voltaic array output time series (p):',np.mean(p), "\n \n")
print(' the avg of abosulte of residual errors (r):',np.mean(r.value),"\n \n")
 
### Plotting  the rsults 
plt.figure()
plt.plot(c.value)
plt.title('Plot clear sky output')
plt.ylabel('clear sky output c')
 
plt.figure()
plt.plot(s.value)
plt.title('plot for wheather shading loss ')
plt.ylabel('shading loss component s')
 
plt.figure()
plt.plot(r.value)
plt.title('plot overal residual error   ')
plt.ylabel('residual errors')
 
plt.figure()
plt.plot(p)
plt.title('plot for power characteristic of  photo voltaic ')
plt.ylabel('PV array output time series')



  
