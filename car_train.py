import glob
import pickle
import cv2
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
def load_dataset():
    car=glob.glob("dataset/vehicles/*.png")
    non_car= glob.glob("dataset/non-vehicles/*.png")
    return car, non_car

def read_img(dataset,pos=True):
    data=[]
    for index,img_path in enumerate(dataset):
        img=cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if pos==True:
            data.append([img,1])
        else:
            data.append([img, 0])
    return np.array(data, dtype=object)
car, non_car= load_dataset()

car_data= read_img(car)
non_car_data= read_img(non_car, pos=False)

shuffle(car_data)
shuffle(non_car_data)

print(len(car_data), len(non_car_data))
training= np.vstack((car_data[:2000],non_car_data[:4000]))
testing = np.vstack((car_data[2000:], non_car_data[4000:]))
with open("car_train.pkl",'wb') as f:
    pickle.dump(training,f)
with open("car_test.pkl",'wb') as f:
    pickle.dump(testing, f)