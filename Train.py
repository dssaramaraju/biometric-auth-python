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
data_1 = os.listdir('Dataset/FingerPrint/Authorized')
  
data_2 = os.listdir('Dataset/FingerPrint/Not')
  


# --- PALMPRINT




data_3 = os.listdir('Dataset/Hand Vein/Authorized')

data_4 = os.listdir('Dataset/Hand Vein/Non')




# --- FINGERPRINT
  
dot1= []
labels1 = []
  
for img in data_1:
      # print(img)
    try:
      img_1 = mpimg.imread('Dataset/FingerPrint/Authorized/' + "/" + img)
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
      img_1 = mpimg.imread('Dataset/FingerPrint/Not/' + "/" + img)
      img_1 = cv2.resize(img_1,((50, 50)))
  
  
  
      try:            
          gray = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
          
      except:
          gray = img_1
  
      
      dot1.append(np.array(gray))
      labels1.append(1)
    except:
        None        




      
import pickle
with open('dot.pickle', 'wb') as f:
    pickle.dump(dot1, f)
    
with open('labels.pickle', 'wb') as f:
    pickle.dump(labels1, f)   
    
    
        
  # --- PALMPRINT     
  
  
dot_palm= []
lab_palm = []
    
  
        
      
for img in data_3:
      # print(img)
    try:
      img_1 = mpimg.imread('Dataset/Hand Vein/Authorized/' + "/" + img)
      img_1 = cv2.resize(img_1,((50, 50)))
  
  
  
      try:            
          gray = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
          
      except:
          gray = img_1
  
      
      dot_palm.append(np.array(gray))
      lab_palm.append(0)      
    except:
        None      
      
for img in data_4:
      # print(img)
    try:
      img_1 = mpimg.imread('Dataset/Hand Vein/Non/' + "/" + img)
      img_1 = cv2.resize(img_1,((50, 50)))
  
  
  
      try:            
          gray = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
          
      except:
          gray = img_1
  
      
      dot_palm.append(np.array(gray))
      lab_palm.append(1)            
      
    except:
        None      
      
 
      
import pickle
with open('dot_hand.pickle', 'wb') as f:
    pickle.dump(dot_palm, f)
    
with open('labels_hand.pickle', 'wb') as f:
    pickle.dump(lab_palm, f)   
    
    
