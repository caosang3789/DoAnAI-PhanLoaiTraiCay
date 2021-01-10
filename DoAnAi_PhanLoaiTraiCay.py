#!/usr/bin/env python
# coding: utf-8

# In[4]:



import numpy as np 
import pandas as pd 
import os
from glob import glob
import matplotlib.pyplot as plt


# In[5]:


# dataset
train_path = '.../Fruit-Images-Dataset-master/Training/'
valid_path = '.../Fruit-Images-Dataset-master/Test/'


# In[6]:


# lấy ảnh từ dataset
image_files = glob(train_path + '/*/*.jp*g')
valid_image_files = glob(valid_path + '/*/*.jp*g')


# In[7]:


# nhận số lớp
folders = glob(train_path + '/*')

# hiển thị 1 ảnh ngẫu nhiên
plt.imshow(plt.imread(np.random.choice(image_files)))
plt.axis('off')
plt.show()


# In[8]:



from keras.layers import Input, Lambda, Dense, Flatten
from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix


# In[9]:


# thay đổi kích thước hình ảnh
IMAGE_SIZE = [100, 100]
# cấu hình training
epochs = 5
batch_size = 32


# In[10]:


vgg = VGG16(input_shape=IMAGE_SIZE + [3], weights='.../Classification-of-Fruit-images-using-VGG16-master/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5', include_top=False)
# không train trọng lượng hiện có
for layer in vgg.layers:
  layer.trainable = False

x = Flatten()(vgg.output)
prediction = Dense(len(folders), activation='softmax')(x)


# In[11]:


# tạo Model
model = Model(inputs=vgg.input, outputs=prediction)

# xem qua cấu trúc model
model.summary()

# định cấu hình model
model.compile(
  loss='categorical_crossentropy',
  optimizer='rmsprop',
  metrics=['accuracy']
)


# In[12]:


# tạo 1 phiên bản ImageDataGenerator
gen = ImageDataGenerator(
  rotation_range=20,
  width_shift_range=0.1,
  height_shift_range=0.1,
  shear_range=0.1,
  zoom_range=0.2,
  horizontal_flip=True,
  vertical_flip=True,
  rescale=1./255,  
  preprocessing_function=preprocess_input
)

# Lấy ánh xạ của class và label number
test_gen = gen.flow_from_directory(valid_path, target_size=IMAGE_SIZE)
print(test_gen.class_indices)
labels = [None] * len(test_gen.class_indices)
for k, v in test_gen.class_indices.items():
  labels[v] = k


# In[13]:


# tạo generators để huấn luyện và xác nhận
train_generator = gen.flow_from_directory(
  train_path,
  target_size=IMAGE_SIZE,
  shuffle=True,
  batch_size=batch_size,
)
valid_generator = gen.flow_from_directory(
  valid_path,
  target_size=IMAGE_SIZE,
  shuffle=False,
  batch_size=batch_size,
)
# Fit the model
r = model.fit_generator(
  train_generator,
  validation_data=valid_generator,
  epochs=epochs,
  steps_per_epoch=len(image_files) // batch_size,
  validation_steps=len(valid_image_files) // batch_size,
)


# In[14]:


# đồ thị sai lệch
plt.plot(r.history['loss'], label='train loss')
plt.plot(r.history['val_loss'], label='val loss')
plt.legend()
plt.show()

# đồ thị thẩm định độ chính xác
plt.plot(r.history['accuracy'], label='train accuracy')
plt.plot(r.history['val_accuracy'], label='val accuracy')
plt.legend()
plt.show()


# In[15]:


print("Final training accuracy = {}".format(r.history["accuracy"][-1]))
print("Final validation accuracy = {}".format(r.history["val_accuracy"][-1]))


# In[17]:


# phỏng đoán
result = np.round(model.predict_generator(valid_generator))
import random
test_files = []
actual_res = []
test_res = []
for i in range(0, 3):
  rng = random.randint(0, len(valid_generator.filenames))
  test_files.append(valid_path +  valid_generator.filenames[rng])
  actual_res.append(valid_generator.filenames[rng].split('/')[0])
  test_res.append(labels[np.argmax(result[rng])])
  
from IPython.display import Image, display
for i in range(0, 3):
  display(Image(test_files[i]))
  print("Actual class: " + str(actual_res[i]))
  print("Predicted class: " + str(test_res[i]))
  


# In[ ]:




