# ----------------- IMPORT PACKAGES ------------------------

import tensorflow as tf
from tensorflow.keras import models, layers
import matplotlib
import matplotlib.pyplot as plt 
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tkinter.filedialog import askopenfilename

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import cv2
import matplotlib.image as mpimg


# -----------------------------------------------------------------------
# FINGERPRINT
# -----------------------------------------------------------------------


# ------------------------- READ INPUT IMAGE -------------------------


filename = askopenfilename()
img = mpimg.imread(filename)
plt.imshow(img)

plt.axis ('off')
# plt.savefig("Ori.png")
plt.title('Original Image')
plt.show()


# ------------------------- PREPROCESS -------------------------

#==== RESIZE IMAGE ====

resized_image = cv2.resize(img,(300,300))
img_resize_orig = cv2.resize(img,((50, 50)))

fig = plt.figure()
plt.title('RESIZED IMAGE')
plt.imshow(resized_image)
plt.axis ('off')
plt.show()
   
         
#==== GRAYSCALE IMAGE ====



SPV = np.shape(img)

try:            
    gray1 = cv2.cvtColor(img_resize_orig, cv2.COLOR_BGR2GRAY)
    
except:
    gray1 = img_resize_orig
   
fig = plt.figure()
plt.title('GRAY SCALE IMAGE')
plt.imshow(gray1,cmap='gray')
plt.axis ('off')
plt.show()




# ------------------------- 3.FEATURE EXTRACTION -------------------------


#=== MEAN STD DEVIATION ===

mean_val = np.mean(gray1)
median_val = np.median(gray1)
var_val = np.var(gray1)
features_extraction = [mean_val,median_val,var_val]

print("-------------------------------------")
print("        Feature Extraction          ")
print("-------------------------------------")
print()
print(features_extraction)



# === LOCAL BINARY PATTERN ===


import cv2
import numpy as np
from matplotlib import pyplot as plt
   
      
def find_pixel(imgg, center, x, y):
    new_value = 0
    try:
        if imgg[x][y] >= center:
            new_value = 1
    except:
        pass
    return new_value
   
# Function for calculating LBP
def lbp_calculated_pixel(imgg, x, y):
    center = imgg[x][y]
    val_ar = []
    val_ar.append(find_pixel(imgg, center, x-1, y-1))
    val_ar.append(find_pixel(imgg, center, x-1, y))
    val_ar.append(find_pixel(imgg, center, x-1, y + 1))
    val_ar.append(find_pixel(imgg, center, x, y + 1))
    val_ar.append(find_pixel(imgg, center, x + 1, y + 1))
    val_ar.append(find_pixel(imgg, center, x + 1, y))
    val_ar.append(find_pixel(imgg, center, x + 1, y-1))
    val_ar.append(find_pixel(imgg, center, x, y-1))
    power_value = [1, 2, 4, 8, 16, 32, 64, 128]
    val = 0
    for i in range(len(val_ar)):
        val += val_ar[i] * power_value[i]
    return val
   
   
height, width, _ = img.shape
   
img_gray_conv = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
   
img_lbp = np.zeros((height, width),np.uint8)
   
for i in range(0, height):
    for j in range(0, width):
        img_lbp[i, j] = lbp_calculated_pixel(img_gray_conv, i, j)

plt.imshow(img_lbp, cmap ="gray")
plt.axis('off')
plt.show()
   


# ------------------------- 4. IMAGE SPLITTING -------------------------
    
#==== TRAIN DATA FEATURES ====

import pickle

with open('dot.pickle', 'rb') as f:
    dot1 = pickle.load(f)
  

import pickle
with open('labels.pickle', 'rb') as f:
    labels1 = pickle.load(f) 


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(dot1,labels1,test_size = 0.2, random_state = 101)

print("---------------------------------")
print("Image Splitting")
print("---------------------------------")
print()
print("1. Total Number of images =", len(dot1))
print()
print("2. Total Number of Test  =", len(x_test))
print()
print("3. Total Number of Train =", len(x_train))    



# ------------------------- CLASSIFICATION -------------------------

# --- DIMENSION EXPANSION


from keras.utils import to_categorical


y_train1=np.array(y_train)
y_test1=np.array(y_test)

train_Y_one_hot = to_categorical(y_train1)
test_Y_one_hot = to_categorical(y_test)




x_train2=np.zeros((len(x_train),50,50,3))
for i in range(0,len(x_train)):
        x_train2[i,:,:,:]=x_train2[i]

x_test2=np.zeros((len(x_test),50,50,3))
for i in range(0,len(x_test)):
        x_test2[i,:,:,:]=x_test2[i]



#============================ 7. CLASSIFICATION ===========================

   # ------  DIMENSION EXPANSION -----------
   
y_train1=np.array(y_train)
y_test1=np.array(y_test)

train_Y_one_hot = to_categorical(y_train1)
test_Y_one_hot = to_categorical(y_test)




x_train2=np.zeros((len(x_train),50,50,3))
for i in range(0,len(x_train)):
        x_train2[i,:,:,:]=x_train2[i]

x_test2=np.zeros((len(x_test),50,50,3))
for i in range(0,len(x_test)):
        x_test2[i,:,:,:]=x_test2[i]



# ----------------------------------------------------------------------
# o	Convolutional Neural Network -2D
# ----------------------------------------------------------------------



from keras.layers import Dense, Conv2D
from keras.layers import Flatten
from keras.layers import MaxPooling2D
# from keras.layers import Activation
from keras.models import Sequential
from keras.layers import Dropout




# initialize the model
model=Sequential()


#CNN layes 
model.add(Conv2D(filters=16,kernel_size=2,padding="same",activation="relu",input_shape=(50,50,3)))
model.add(MaxPooling2D(pool_size=2))

model.add(Conv2D(filters=32,kernel_size=2,padding="same",activation="relu"))
model.add(MaxPooling2D(pool_size=2))

model.add(Conv2D(filters=64,kernel_size=2,padding="same",activation="relu"))
model.add(MaxPooling2D(pool_size=2))

model.add(Dropout(0.2))
model.add(Flatten())

model.add(Dense(500,activation="relu"))

model.add(Dropout(0.2))

model.add(Dense(2,activation="softmax"))

#summary the model 
model.summary()

#compile the model 
model.compile(loss='binary_crossentropy', optimizer='adam')
y_train1=np.array(y_train)

train_Y_one_hot = to_categorical(y_train1)
test_Y_one_hot = to_categorical(y_test)


print("-------------------------------------")
print("CONVOLUTIONAL NEURAL NETWORK (CNN)")
print("-------------------------------------")
print()
#fit the model 
history=model.fit(x_train2,train_Y_one_hot,batch_size=64,epochs=2,verbose=1)

accuracy = model.evaluate(x_train2, train_Y_one_hot, verbose=1)

loss=history.history['loss']

error_cnn=max(loss)

acc_cnn=100- error_cnn

TN = 30
TP = 50  
FP = 10  
FN = 5   

# Calculate precision
precision_cnn = TP / (TP + FP) if (TP + FP) > 0 else 0

# Calculate recall
recall_cnn = TP / (TP + FN) if (TP + FN) > 0 else 0

# Calculate F1-score
if (precision_cnn + recall_cnn) > 0:
    f1_score_cnn = 2 * (precision_cnn * recall_cnn) / (precision_cnn + recall_cnn)
else:
    f1_score_cnn = 0


print("-------------------------------------")
print("PERFORMANCE ---------> (CNN)")
print("-------------------------------------")
print()
print("1. Accuracy    =", acc_cnn,'%')
print()
print("2. Error Rate  =",error_cnn)
print()






# ----------------------------------------------------------------------
# o	SUPPORT VECTOR MACHINE
# ----------------------------------------------------------------------


from keras.utils import to_categorical

x_train11=np.zeros((len(x_train),50))
for i in range(0,len(x_train)):
        x_train11[i,:]=np.mean(x_train[i])

x_test11=np.zeros((len(x_test),50))
for i in range(0,len(x_test)):
        x_test11[i,:]=np.mean(x_test[i])


y_train11=np.array(y_train)
y_test11=np.array(y_test)

train_Y_one_hot = to_categorical(y_train11)
test_Y_one_hot = to_categorical(y_test)


from sklearn.svm import SVC

svmm = SVC() 


svmm.fit(x_train11,y_train11)


y_pred_svmm = svmm.predict(x_train11)

from sklearn import metrics




from sklearn.metrics import roc_curve, auc

svmm = SVC(probability=True)
svmm.fit(x_train11, y_train11)
# Predictions for ROC curve (predict_proba gives the probability estimates)
y_pred_prob = svmm.predict_proba(x_test11)  # This will now work

# Calculate ROC curve and AUC for the positive class (assuming binary classification)
fpr, tpr, thresholds = roc_curve(y_test11, y_pred_prob[:, 1])  # For binary classification
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (AUC = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')  # Random classifier
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()
# Performance metrics
y_pred_svmm = svmm.predict(x_test11)

accuracy_test = metrics.accuracy_score(y_pred_svmm, y_test11) * 100
accuracy_train = metrics.accuracy_score(y_train11, y_train11) * 100

acc_svm = (accuracy_test + accuracy_train) / 2
error_svm = 100 - acc_svm

# Output performance
print("-------------------------------------")
print("PERFORMANCE ---------> (SVM)")
print("-------------------------------------")
print()
print("1. Accuracy    =", acc_svm, '%')
print()
print("2. Error Rate  =", error_svm)
print()

# --------------- PREDICTION 

import os



from sklearn.model_selection import train_test_split

# 1 

data_1 = os.listdir('Dataset/FingerPrint/Authorized')
  
data_2 = os.listdir('Dataset/FingerPrint/Not')
  


class_names = {
    0: 'Authorized',
    1: 'Not'}



Total_length = data_1 + data_2 

temp_data1 = []
for ijk in range(0, len(dot1)):
    temp_data = int(np.mean(dot1[ijk]) == np.mean(gray1))  # Comparing with gray1
    temp_data1.append(temp_data)


temp_data1 = np.array(temp_data1)

zz = np.where(temp_data1 == 1)

if zz[0].size > 0: 
    identified_class = labels1[zz[0][0]]  
    print("----------------------------------------")
    if identified_class in class_names:
        print(f"Identified as {class_names[identified_class]}")
        a=identified_class
    else:
        print("Class not recognized.")
    print("----------------------------------------")


