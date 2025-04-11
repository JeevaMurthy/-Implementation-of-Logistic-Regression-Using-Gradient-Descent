# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the necessary python packages.
2. Read the dataset.
3. Define X and Y array
4. Define a function for costFunction,cost and gradient.
5. Define a function to plot the decision boundary and predict the Regression value.

## Program & Output:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: JEEVA K
RegisterNumber:  212223230090
*/
```
```pyton
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('Placement_Data.csv')
dataset

```

![image](https://github.com/user-attachments/assets/c77f855a-bece-41e2-afa6-d8adef95a3f7)

```pyton
dataset = dataset.drop('sl_no',axis=1)
dataset = dataset.drop('salary',axis=1)

```


```pyton
dataset["gender"]=dataset["gender"].astype('category')
dataset["ssc_b"]=dataset["ssc_b"].astype('category')
dataset["hsc_b"]=dataset["hsc_b"].astype('category')
dataset["degree_t"]=dataset["degree_t"].astype('category')
dataset["workex"]=dataset["workex"].astype('category')
dataset["specialisation"]=dataset["specialisation"].astype('category')
dataset["status"]=dataset["status"].astype('category')
dataset["hsc_s"]=dataset["hsc_s"].astype('category')
dataset.dtypes

```

![image](https://github.com/user-attachments/assets/ca163106-ed7f-474f-a9c9-b15c8f229080)


```pyton
dataset["gender"]=dataset["gender"].cat.codes
dataset["ssc_b"]=dataset["ssc_b"].cat.codes
dataset["hsc_b"]=dataset["hsc_b"].cat.codes
dataset["degree_t"]=dataset["degree_t"].cat.codes
dataset["workex"]=dataset["workex"].cat.codes
dataset["specialisation"]=dataset["specialisation"].cat.codes
dataset["status"]=dataset["status"].cat.codes
dataset["hsc_s"]=dataset["hsc_s"].cat.codes
dataset

```

![image](https://github.com/user-attachments/assets/1218db0e-3800-4db0-95dd-7ddc10045799)


```pyton
X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,-1].values
Y
```

![Screenshot 2025-04-11 090944](https://github.com/user-attachments/assets/1361846d-0877-47e9-9e7b-9799f899fa64)


```pyton
theta = np.random.randn(X.shape[1])
y =Y
def sigmoid(z):
    return 1/(1+np.exp(-z))
def loss(theta,X,y):
    h= sigmoid(X.dot(theta))
    return -np.sum(y*np.log(h)+(1-y)*np.log(1-h))

def gradient_descent(theta,X,y,alpha,num_iterations):
    m = len(y)
    for i in range(num_iterations):
        h = sigmoid(X.dot(theta))
        gradient = X.T.dot(h-y)/m
        theta -= alpha*gradient
    return theta

theta = gradient_descent(theta,X,y,alpha=0.01,num_iterations = 1000)

```


```pyton
def predict(theta,X):
    h = sigmoid(X.dot(theta))
    y_pred = np.where(h>=0.5,1,0)
    return y_pred

y_pred = predict(theta,X)

```

```pyton
accuracy = np.mean(y_pred.flatten()==y)
print("Accuracy:", accuracy)

```
![image](https://github.com/user-attachments/assets/290e1e05-de39-4dc4-ba6f-d8d19dfa3a9a)

```pyton
print(y_pred)
```
![image](https://github.com/user-attachments/assets/a24cb364-f0e1-424f-b91d-988b36e28e56)

```pyton
print(Y)
```
![image](https://github.com/user-attachments/assets/12d60b56-f2dd-4aa1-89a2-eeef66d63ec7)

```python
xnew = np.array([[0,87,0,95,0,2,78,2,0,0,1,0]])
y_prednew = predict(theta,xnew)
print(y_prednew)

```
![image](https://github.com/user-attachments/assets/3590a784-69f3-4cde-a5bc-03cc46022201)

```python
xnew = np.array([[0,0,0,0,0,2,8,2,0,0,1,0]])
y_prednew = predict(theta,xnew)
print(y_prednew) 
```
![image](https://github.com/user-attachments/assets/84ee4959-33c3-4df6-bee2-6ad12d0709ad)

## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

