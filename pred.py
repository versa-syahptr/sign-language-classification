import numpy as np
import joblib
import cv2
from PIL import Image


# lgbm_model = joblib.load('tubes-pcd-LightGBM.pkl')
# svm_model = joblib.load('tubes-pcd-SVM.pkl')

def apply_clahe(image, clip_limit=2.0, grid_size=(8, 8)):

    '''
    image = array of 2D image
    '''

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    return clahe.apply(image)

def model_prediction(uploaded_image, model, use_clahe=True):

    '''
    uploaded_image: str = image path
    model_name: str = lgbm or svm
    use_clahe: bool = either or not using clahe
    '''

    # CLAHE preprocessing

    # img = Image.open(uploaded_image)
    # img = img.resize((28, 28))
    # img = np.array(img)
    if type(uploaded_image) == Image.Image:
        img = np.array(uploaded_image)
    else:
        img = uploaded_image
        
    if use_clahe:
        img = apply_clahe(img, clip_limit=8.0, grid_size=(2, 2))
    descriptors = img.flatten()

    
    descriptors = np.array(descriptors).reshape(1, -1)

    # if model_name == 'lgbm':
    #     model = lgbm_model
    # elif model_name == 'svm':
    #     model = svm_model
    prediction = model.predict(descriptors)
    
    return prediction