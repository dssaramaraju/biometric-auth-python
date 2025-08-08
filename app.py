# ================ Input Packages =========================

import streamlit as st

import base64
import matplotlib.image as mpimg
import cv2

import matplotlib.pyplot as plt 
from streamlit_option_menu import option_menu


# ================ Background image ==============================


st.markdown(f'<h1 style="color:#000000;text-align: center;font-size:36px;font-family:concat">{"Multimodal fusion based person authentication using deep learning"}</h1>', unsafe_allow_html=True)


def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
        background-size: cover
    }}
    </style>
    """,
    unsafe_allow_html=True
    )
add_bg_from_local('1.avif')



selected = option_menu(
    menu_title=None, 
    options=["Fingerprint","Hand Vein","Overall Result"],  
    orientation="horizontal",
)


st.markdown(
    """
    <style>
    .option_menu_container {
        position: fixed;
        top: 20px;
        right: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)


if selected == "Fingerprint":
    
    
       
    #====================== READ A INPUT IMAGE =========================
    
    fileneme = st.file_uploader("Upload a image")
    
    if fileneme is None:
        
        st.text("Kindly upload input image....")
    
    else:
        # selected_image_name = fileneme.name
        #====================== READ A INPUT IMAGE =========================
        
        # from tkinter.filedialog import askopenfilename
        # fileneme = askopenfilename()
        
        col1,col2,col3 = st.columns(3)
        
        with col2:
        
            img = mpimg.imread(fileneme)
            plt.imshow(img)
            plt.axis ('off')
            plt.savefig("Ori.png")
            plt.show()
            
            
            st.image(img,caption='Original Image')
        
        
        #============================ PREPROCESS =================================
        
        #==== RESIZE IMAGE ====
        
        resized_image = cv2.resize(img,(300,300))
        img_resize_orig = cv2.resize(img,((50, 50)))
        
    
                 
        #==== GRAYSCALE IMAGE ====
        
        
        
        try:            
            gray1 = cv2.cvtColor(img_resize_orig, cv2.COLOR_BGR2GRAY)
            
        except:
            gray1 = img_resize_orig
           
    
        
        
        # ------------------------- 4. IMAGE SPLITTING -------------------------
        
        
            
        #==== TRAIN DATA FEATURES ====
        
        import pickle
        
        with open('fin_d.pickle', 'rb') as f:
            dot1 = pickle.load(f)
          
        
        import pickle
        with open('fin_l.pickle', 'rb') as f:
            labels1 = pickle.load(f)     
        
        
        # --------------------------------- PREDICTION ----------------------------
        
        
        import os
        import numpy as np
        
        
        from sklearn.model_selection import train_test_split
        
         # 1 
        
     
        
        temp_data1 = []
        for ijk in range(0, len(dot1)):
             temp_data = int(np.mean(dot1[ijk]) == np.mean(gray1))  # Comparing with gray1
             temp_data1.append(temp_data)
        
        
        temp_data1 = np.array(temp_data1)
        
        zz = np.where(temp_data1 == 1)
    
        if zz[0].size > 0: 
            identified_class = labels1[zz[0][0]]  
            print("----------------------------------------")
            # if identified_class in class_names:
            #     print(f"Identified Palmprint as {class_names[identified_class]}")
            #     aa = "Identified fingerprint is  " + class_names[identified_class]
            #     st.markdown(f'<h1 style="color:#000000;text-align: center;font-size:30px;font-family:Caveat, sans-serif;">{aa}</h1>', unsafe_allow_html=True)
            #     import pickle
            #     finger=class_names[identified_class]
            with open('c1.pickle', 'wb') as f:
                pickle.dump(identified_class, f)
            st.success("Uploaded Succesfully")

            
            print("----------------------------------------")
            
        
        

if selected == "Hand Vein":
    
    
       
    #====================== READ A INPUT IMAGE =========================
    
    fileneme = st.file_uploader("Upload a image")
    
    if fileneme is None:
        
        st.text("Kindly upload input image....")
    
    else:
        # selected_image_name = fileneme.name
        #====================== READ A INPUT IMAGE =========================
        
        # from tkinter.filedialog import askopenfilename
        # fileneme = askopenfilename()
        
        col1,col2,col3 = st.columns(3)
        
        with col2:
        
            img = mpimg.imread(fileneme)
            plt.imshow(img)
            plt.axis ('off')
            plt.savefig("Ori.png")
            plt.show()
            
            
            st.image(img,caption='Original Image')
        
        
        #============================ PREPROCESS =================================
        
        #==== RESIZE IMAGE ====
        
        resized_image = cv2.resize(img,(300,300))
        img_resize_orig = cv2.resize(img,((50, 50)))
        
    
                 
        #==== GRAYSCALE IMAGE ====
        
        
        
        try:            
            gray1 = cv2.cvtColor(img_resize_orig, cv2.COLOR_BGR2GRAY)
            
        except:
            gray1 = img_resize_orig
           
    
        
        
        # ------------------------- 4. IMAGE SPLITTING -------------------------
        
        
            
        #==== TRAIN DATA FEATURES ====
        
        import pickle
        
        with open('fin_d.pickle', 'rb') as f:
            dot1 = pickle.load(f)
          
        
        import pickle
        with open('fin_l.pickle', 'rb') as f:
            labels1 = pickle.load(f)     
        
        
        # --------------------------------- PREDICTION ----------------------------
        
        
        import os
        import numpy as np
        
        
        from sklearn.model_selection import train_test_split
        
         # 1 
        
     
        
        temp_data1 = []
        for ijk in range(0, len(dot1)):
             temp_data = int(np.mean(dot1[ijk]) == np.mean(gray1))  # Comparing with gray1
             temp_data1.append(temp_data)
        
        
        temp_data1 = np.array(temp_data1)
        
        zz = np.where(temp_data1 == 1)
    
        if zz[0].size > 0: 
            identified_class = labels1[zz[0][0]]  
            print("----------------------------------------")
            # if identified_class in class_names:
            #     print(f"Identified Palmprint as {class_names[identified_class]}")
            #     aa = "Identified fingerprint is  " + class_names[identified_class]
            #     st.markdown(f'<h1 style="color:#000000;text-align: center;font-size:30px;font-family:Caveat, sans-serif;">{aa}</h1>', unsafe_allow_html=True)
            #     import pickle
            #     finger=class_names[identified_class]
            with open('c2.pickle', 'wb') as f:
                pickle.dump(identified_class, f)
            st.success("Uploaded Succesfully")
            
            print("----------------------------------------")
            
            
            
if selected=="Overall Result":


            
  import pickle

  with open('c1.pickle', 'rb') as f:
      res1 = pickle.load(f)    
            
  with open('c2.pickle', 'rb') as f:
      res2 = pickle.load(f) 
      
  if int(res1) == int(res2):
      
      print("Authorized")
      st.markdown(f'<h1 style="color:#000000;text-align: center;font-size:30px;font-family:Caveat, sans-serif;">{"Identified Person is Authorized"}</h1>', unsafe_allow_html=True)

  else:
      st.markdown(f'<h1 style="color:#000000;text-align: center;font-size:30px;font-family:Caveat, sans-serif;">{"Identified Person is Un-Authorized"}</h1>', unsafe_allow_html=True)
      
      print("Un-Authorized")
      
     

        
      
        
      
        
      
        
      
        
      
        
      
            
