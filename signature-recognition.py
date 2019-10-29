#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import os 
import cv2

datadir="E:\data 2"
categories=["ain","al","aleff","bb","dal","dha","dhad","fa","gaaf","ghain","ha","haa","jeem","kaaf","la","laam","meem","nun","ra","saad","seen","sheen","ta","taa","thaa","thal","toot","waw","ya","yaa","zay"]
len(categories)


# In[2]:


for category in categories:
    path=os.path.join(datadir,category)
    for img in os.listdir(path):
        img_array=cv2.imread(os.path.join(path,img),cv2.IMREAD_GRAYSCALE)
        plt.imshow(img_array,cmap="gray")
        plt.show()
        break
    break


# In[3]:


print(img_array.shape)


# In[4]:


img_size=60
# to make the every photo at the same size
new_array=cv2.resize(img_array,(img_size,img_size))
plt.imshow(new_array,cmap='gray')
plt.show()


# In[5]:


training_data=[]
def create_training_data():
    for category in categories:  

        path = os.path.join(datadir,category)  
        class_num = categories.index(category)  

        for img in os.listdir(path):  
            try:
                img_array = cv2.imread(os.path.join(path,img) ,cv2.IMREAD_GRAYSCALE) 
                new_array = cv2.resize(img_array, (img_size, img_size))  
                training_data.append([new_array, class_num]) 
            except Exception as e:  
                pass
            

create_training_data()


# In[6]:


print(len(training_data))


# In[7]:


import random
random.shuffle(training_data)
for sample in training_data[:10]:
    print(sample[1])


# In[8]:


X =[]
y =[]
for features,label in training_data:
    X.append(features)
    y.append(label)
    
  

X = np.array(X).reshape(-1,img_size,img_size,1)


# In[17]:


X.shape


# In[18]:


z=np.array(y)
z.shape


# In[9]:


import tensorflow as tf
from tensorflow import keras
model = keras.Sequential([
    keras.layers.Flatten(input_shape = (img_size,img_size,1)),
    keras.layers.Dense(180, activation = tf.nn.relu),
    keras.layers.Dense(180, activation = tf.nn.relu),
    keras.layers.Dense(180, activation = tf.nn.relu),
    keras.layers.Dense(180, activation = tf.nn.relu),
    keras.layers.Dense(180, activation = tf.nn.relu),
    keras.layers.Dense(180, activation = tf.nn.relu),
    keras.layers.Dense(180, activation = tf.nn.relu),
    keras.layers.Dense(180, activation = tf.nn.relu),
    keras.layers.Dense(180, activation = tf.nn.softmax)
])
model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy',metrics = ['accuracy'])


# In[14]:


model.fit(X,y,epochs=15) # run this code 3 times to get this accuracy


# In[15]:


test_loss , test_acc = model.evaluate(X,y)


# In[16]:


print("the accuracy is : ",test_acc)


# In[ ]:





# In[ ]:




