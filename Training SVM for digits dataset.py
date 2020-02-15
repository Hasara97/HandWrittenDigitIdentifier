from sklearn.datasets import load_digits
import cv2

dataset=load_digits()

data=dataset.data
target=dataset.target
#imgs=dataset.images


ret,data=cv2.threshold(data,7,15,cv2.THRESH_BINARY_INV)
#thresholding the whole dataset at once

from sklearn.svm import SVC

algorithm=SVC()

algorithm.fit(data,target)

import joblib

joblib.dump(algorithm,'digits_model_svm.sav')
