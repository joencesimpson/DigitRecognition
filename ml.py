import numpy as np
import pandas as pd
import matplotlib.pyplot as pt
from matplotlib.image import imread
from PIL import Image
from PIL import Image, ImageOps
from sklearn.tree import DecisionTreeClassifier

#importuing the Dataset
data=pd.read_csv("C:/Users/Joence/Downloads/Mini_project/train.csv").to_numpy()

# training dataset
clf=DecisionTreeClassifier()
xtrain=data[0:41600, 1:]
train_label=data[0:41600,0]
clf.fit(xtrain, train_label)

#image convertion
img = Image.open("C:/Users/Joence/Downloads/Mini_project/test9.png")
img1=img.resize((28,28))
img2=img1.convert('L')
imgar=np.array(img2)
img3= np.reshape(imgar,(784))

# # testing data
# xtest=data[41600:,1:]
# actual_label=data[41600:,0]
# d=xtest[90]
# d.shape = (28,28)
# pt.imshow(255-d, cmap='gray')
# ans = clf.predict([xtest[90]])

#Prediction for input image
pt.imshow(img2,cmap ='gray')
ans = clf.predict([img3])

#printing value
print("The predicted digit is :")
print(ans[0])

#plotting graph
pt.show()

# #accuracy verification
# p=clf.predict(xtest)
# cc=0
# for i in range (0,400):
#     if(p[i]==actual_label[i]):
#         cc=cc+1
# acc=(cc/400)*100
# print("Accuracy=",acc,"%")
