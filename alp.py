import numpy as np
import pandas as pd
import matplotlib.pyplot as pt
from matplotlib.image import imread
from PIL import Image
from PIL import Image, ImageOps
from sklearn.tree import DecisionTreeClassifier

#number to alphabet
def alphie(argument):
    switcher = {
        0: "a",1: "b",2: "c",3:'d',4:'e',5:'f',6:'g',7:'h',
    8:'i',9:'j',10:'k',11:'l',12:'m',13:'n',14:'o',15:'p',16:'q',
    17:'r',18:'s',19:'t',20:'u',21:'v',22:'w',23:'x',24:'y',25:'z'
    }

    return switcher.get(argument, "nothing")

#importing the dataset
data=pd.read_csv("F:/Downloads/Mini_project/A_Z Handwritten Data.csv").to_numpy()

#image input
img = Image.open("F:/Downloads/Mini_project/test111.png")
img1=img.resize((28,28))
img2=img1.convert('L')
imgar=np.array(img2)
img3= np.reshape(imgar,(784))

# training dataset
clf=DecisionTreeClassifier()
xtrain=data[0:372050, 1:]
train_label=data[0:372050,0]
clf.fit(xtrain, train_label)

# # testing data
# xtest=data[372050:,1:]
# actual_label=data[372050:,0]
# d=xtest[40]
# d.shape = (28,28)
# pt.imshow(255-d, cmap='gray')
# ans =clf.predict([xtest[40]])

#Prediction for input image
pt.imshow(img2,cmap ='gray')
ans= clf.predict([img3])

#printing value
print("The predicted alphabet is :")
ans1 = ans[0]
ans3 = alphie(ans1)
print(ans3)

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
