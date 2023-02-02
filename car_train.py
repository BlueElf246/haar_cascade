import glob
import pickle
import cv2
import numpy as np
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

training= np.vstack((car_data,non_car_data))

with open("car_train.pkl",'wb') as f:
    pickle.dump(training,f)