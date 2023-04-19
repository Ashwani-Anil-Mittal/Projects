#!/usr/bin/env python
# coding: utf-8

# Rosenbrock Function

# In[19]:


import numpy as np
import matplotlib.pyplot as plt

#calulate rosenbrock function
def Rosenbrock(x,y):
    return (1 - x)**2 + 100*(y - x**2)**2

#find the gradients
def Grad_Rosenbrock(x,y):
    g1 = -400*x*y + 400*x**3 + 2*x -2
    g2 = 200*y -200*x**2
    return np.array([g1,g2])


def Gradient_Descent(Grad,x,y, gamma = 0.0009, epsilon=0.00001, nMax = 100000, momentum = 0.5):
    #Initialization
    i = 0
    iter_x, iter_y, iter_count = np.empty(0),np.empty(0), np.empty(0)
    error = 10
    X = np.array([x,y])
    #initial value of change set to 0
    change = 0.0
    #Looping as long as error is greater than epsilon
    while np.linalg.norm(error) > epsilon and i < nMax:
        i +=1
        iter_x = np.append(iter_x,x)
        iter_y = np.append(iter_y,y)
        iter_count = np.append(iter_count ,i)   
        
        X_prev = X
        #Calculate new change value based on momentum factor
        new_change = gamma * Grad(x,y) + momentum * change
        X = X - new_change
        error = X - X_prev
        #Store new change value for next iteration
        change = new_change
        x,y = X[0], X[1]
          
    print(X)
    return X, iter_x,iter_y, iter_count


root,iter_x,iter_y, iter_count = Gradient_Descent(Grad_Rosenbrock,-1,1)
("")
x = np.linspace(-2,2,250)
y = np.linspace(-1,3,250)
X, Y = np.meshgrid(x, y)
Z = Rosenbrock(X, Y)

#Angles needed for quiver plot
anglesx = iter_x[1:] - iter_x[:-1]
anglesy = iter_y[1:] - iter_y[:-1]



fig = plt.figure(figsize = (16,8))



#Contour plot
ax = fig.add_subplot(1, 2, 2)
ax.contour(X,Y,Z, 50, cmap = 'jet')
#Plotting the iterations and intermediate values
ax.scatter(iter_x,iter_y,color = 'r', marker = '*')
ax.quiver(iter_x[:-1], iter_y[:-1], anglesx, anglesy, scale_units = 'xy', angles = 'xy', scale = 1, color = 'r', alpha = .3)
ax.set_title('Gradient Descent with {} iterations'.format(len(iter_count)))


plt.show()


# SPHERE FUNCTION

# In[25]:


import numpy as np
import matplotlib.pyplot as plt

#calulate sphere function
def Sphere(x,y):
    return x**2 + y**2

#find the gradients
def Grad_Sphere(x,y):
    g1 = 2*x 
    g2 = 2*y 
    return np.array([g1,g2])


def Gradient_Descent(Grad,x,y, gamma = 0.0009, epsilon=0.00001, nMax = 10000, momentum = 0.5):
    #Initialization
    i = 0
    iter_x, iter_y, iter_count = np.empty(0),np.empty(0), np.empty(0)
    error = 10
    X = np.array([x,y])
    #initial value of change set to 0
    change = 0.0
    #Looping as long as error is greater than epsilon
    while np.linalg.norm(error) > epsilon and i < nMax:
        i +=1
        iter_x = np.append(iter_x,x)
        iter_y = np.append(iter_y,y)
        iter_count = np.append(iter_count ,i)   
        
        X_prev = X
        #Calculate new change value based on momentum factor
        new_change = gamma * Grad(x,y) + momentum * change
        X = X - new_change
        error = X - X_prev
        #Store new change value for next iteration
        change = new_change
        x,y = X[0], X[1]
    print(X)
    return X, iter_x,iter_y, iter_count




root,iter_x,iter_y, iter_count = Gradient_Descent(Grad_Sphere,-2,2)

x = np.linspace(-2,2,250)
y = np.linspace(-1,3,250)
X, Y = np.meshgrid(x, y)
Z = Sphere(X, Y)

#Angles needed for quiver plot
anglesx = iter_x[1:] - iter_x[:-1]
anglesy = iter_y[1:] - iter_y[:-1]


get_ipython().run_line_magic('matplotlib', 'inline')
fig = plt.figure(figsize = (16,8))


#Contour plot
ax = fig.add_subplot(1, 2, 2)
ax.contour(X,Y,Z, 50, cmap = 'jet')
#Plotting the iterations and intermediate values
ax.scatter(iter_x,iter_y,color = 'r', marker = '*')
ax.quiver(iter_x[:-1], iter_y[:-1], anglesx, anglesy, scale_units = 'xy', angles = 'xy', scale = 1, color = 'r', alpha = .3)
ax.set_title('Gradient Descent with {} iterations'.format(len(iter_count)))


plt.show()


# BOOTH FUNCTION

# In[29]:


import numpy as np
import matplotlib.pyplot as plt

#calulate booth function
def Booth(x,y):
    return (x + 2*y - 7)**2 + (2*x + y - 5)**2

#find the gradients
def Grad_Booth(x,y):
    g1 = 10*x + 8*y - 34
    g2 = 8*x + 10*y - 38
    return np.array([g1,g2])

def Gradient_Descent(Grad,x,y, gamma = 0.0008, epsilon=0.00001, nMax = 10000, momentum = 0.5):
    #Initialization
    i = 0
    iter_x, iter_y, iter_count = np.empty(0),np.empty(0), np.empty(0)
    error = 10
    X = np.array([x,y])
    #initial value of change set to 0
    change = 0.0
    #Looping as long as error is greater than epsilon
    while np.linalg.norm(error) > epsilon and i < nMax:
        i +=1
        iter_x = np.append(iter_x,x)
        iter_y = np.append(iter_y,y)
        iter_count = np.append(iter_count ,i)   
        
        X_prev = X
        #Calculate new change value based on momentum factor
        new_change = gamma * Grad(x,y) + momentum * change
        X = X - new_change
        error = X - X_prev
        #Store new change value for next iteration
        change = new_change
        x,y = X[0], X[1]
    print(X)
    return X, iter_x,iter_y, iter_count

root,iter_x,iter_y, iter_count = Gradient_Descent(Grad_Booth,-8,8)

x = np.linspace(-10,10,1000)
y = np.linspace(-10,10,1000)
X, Y = np.meshgrid(x, y)
Z = Booth(X, Y)

#Angles needed for quiver plot
anglesx = iter_x[1:] - iter_x[:-1]
anglesy = iter_y[1:] - iter_y[:-1]


get_ipython().run_line_magic('matplotlib', 'inline')
fig = plt.figure(figsize = (16,8))



#Contour plot
ax = fig.add_subplot(1, 2, 2)
ax.contour(X,Y,Z, 50, cmap = 'jet')
#Plotting the iterations and intermediate values
ax.scatter(iter_x,iter_y,color = 'r', marker = '*')
ax.quiver(iter_x[:-1], iter_y[:-1], anglesx, anglesy, scale_units = 'xy', angles = 'xy', scale = 1, color = 'r', alpha = .3)
ax.set_title('Gradient Descent with {} iterations'.format(len(iter_count)))


plt.show()

