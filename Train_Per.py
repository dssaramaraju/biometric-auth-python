# ====================== IMPORT PACKAGES ======================
    
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import cv2
import matplotlib.image as mpimg

# --- FINGERPRINT


from sklearn.model_selection import train_test_split

# 1 
data_1 = os.listdir('Data/P1')
  
data_2 = os.listdir('Data/P2')
  
data_3 = os.listdir('Data/P3')

data_4 = os.listdir('Data/P4')

data_5 = os.listdir('Data/P5')
  
data_6 = os.listdir('Data/P6')
  





# --- FINGERPRINT
  
dot1= []
labels1 = []
  
for img in data_1:
      # print(img)
    try:
      img_1 = mpimg.imread('Data/P1/' + "/" + img)
      img_1 = cv2.resize(img_1,((50, 50)))
  
  
  
      try:            
          gray = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
          
      except:
          gray = img_1
  
      
      dot1.append(np.array(gray))
      labels1.append(0)
    except:
        None
        
        
for img in data_2:
      # print(img)
    try:
      img_1 = mpimg.imread('Data/P2/' + "/" + img)
      img_1 = cv2.resize(img_1,((50, 50)))
  
  
  
      try:            
          gray = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
          
      except:
          gray = img_1
  
      
      dot1.append(np.array(gray))
      labels1.append(1)
    except:
        None        

  
        
      
for img in data_3:
      # print(img)
    try:
      img_1 = mpimg.imread('Data/P3/' + "/" + img)
      img_1 = cv2.resize(img_1,((50, 50)))
  
  
  
      try:            
          gray = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
          
      except:
          gray = img_1
  
      
      dot1.append(np.array(gray))
      labels1.append(2)      
    except:
        None      
      
for img in data_4:
      # print(img)
    try:
      img_1 = mpimg.imread('Data/P4/' + "/" + img)
      img_1 = cv2.resize(img_1,((50, 50)))
  
  
  
      try:            
          gray = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
          
      except:
          gray = img_1
  
      
      dot1.append(np.array(gray))
      labels1.append(3)            
      
    except:
        None      
      
 

for img in data_5:
      # print(img)
    try:
      img_1 = mpimg.imread('Data/P5/' + "/" + img)
      img_1 = cv2.resize(img_1,((50, 50)))
  
  
  
      try:            
          gray = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
          
      except:
          gray = img_1
  
      
      dot1.append(np.array(gray))
      labels1.append(4)            
      
    except:
        None      
      


for img in data_6:
      # print(img)
    try:
      img_1 = mpimg.imread('Data/P6/' + "/" + img)
      img_1 = cv2.resize(img_1,((50, 50)))
  
  
  
      try:            
          gray = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
          
      except:
          gray = img_1
  
      
      dot1.append(np.array(gray))
      labels1.append(5)            
      
    except:
        None   



import pickle
with open('fin_d.pickle', 'wb') as f:
    pickle.dump(dot1, f)
    
with open('fin_l.pickle', 'wb') as f:
    pickle.dump(labels1, f)   