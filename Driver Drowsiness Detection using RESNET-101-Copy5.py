#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Importing Libraries
import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.models import Sequential


# In[3]:


import zipfile as zf
files = zf.ZipFile("Eyedataset.zip", 'r')
files.extractall('Eyedataset')
files.close()


# In[2]:


#DataPreprocessing
from tensorflow.keras.preprocessing.image import ImageDataGenerator
data_dir = "C:/Users/TEJASWI/OneDrive/Documents/SE Project/Eyedataset"
train_datagen = ImageDataGenerator(validation_split=0.2)

train_generator = train_datagen.flow_from_directory( data_dir,
                                                     target_size=(32,32),
                                                     batch_size=32,
                                                     shuffle=True,
                                                     class_mode='categorical',
                                                     subset='training')

validation_datagen = ImageDataGenerator(validation_split=0.2)

validation_generator =  validation_datagen.flow_from_directory( data_dir,
                                                                target_size=(32,32),
                                                                batch_size=32,
                                                                class_mode='categorical',
                                                                subset='validation')   


# In[3]:


resnet101_model = Sequential()

pretrained_model = tf.keras.applications.ResNet101(include_top=False,
                                                   input_shape=(32, 32, 3),
                                                   pooling='avg',
                                                   classes=15,
                                                   weights='imagenet')

for layer in pretrained_model.layers:
    layer.trainable = False

resnet101_model.add(pretrained_model)
resnet101_model.add(Flatten())
resnet101_model.add(Dense(128, activation='relu'))
resnet101_model.add(Dense(15, activation='softmax'))


# In[4]:


from tensorflow.keras.optimizers import Adam
resnet101_model.compile(optimizer=Adam(learning_rate=0.001),
                        loss='categorical_crossentropy',
                        metrics=[tf.keras.metrics.CategoricalAccuracy(),
                                 tf.keras.metrics.Precision(),
                                 tf.keras.metrics.Recall()])


# In[5]:


resnet101_model.summary()


# In[6]:


history = resnet101_model.fit(train_generator,
                                validation_data=validation_generator,
                               epochs=50)
   


# In[7]:


fig1 = plt.gcf()
plt.plot(history.history['categorical_accuracy'])
plt.plot(history.history['val_categorical_accuracy'])
plt.axis(ymin=0.4,ymax=1)
plt.grid()
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend(['train', 'validation'])
plt.show()


# In[8]:


import cv2
import numpy as np
img_height, img_width = 32,32
image=cv2.imread(r"C:\Users\TEJASWI\OneDrive\Documents\SE Project\Eyedataset\s0001\s0001_00012_0_0_0_0_0_01.png")
image_resized= cv2.resize(image, (img_height,img_width))
image=np.expand_dims(image_resized,axis=0)


# In[3]:


pip install opencv-python


# In[9]:


import cv2
img_height, img_width = 32,32
image=cv2.imread(r"C:\Users\TEJASWI\OneDrive\Documents\SE Project\Eyedataset\s0001\s0001_00012_0_0_0_0_0_01.png")
print(image.shape)
r_eye = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
print(r_eye.shape)
r_eye = cv2.resize(r_eye, (img_height,img_width))
print(r_eye.shape)
#r_eye= r_eye/255
r_eye=  r_eye.reshape((-1, 32, 32, 3))
print(r_eye.shape)
#r_eye = np.expand_dims(r_eye,axis=0)
pred=resnet101_model.predict(r_eye)


# In[10]:


np.argmax(pred)


# In[11]:


resnet101_model.save("./resnet101.h5")


# In[ ]:




