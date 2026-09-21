import cv2

import numpy as np



def preprocessing_roi(chest_roi):

    if chest_roi is None  or chest_roi.size==0:

        return None, {}

   

    grayscale_step = cv2.cvtColor(chest_roi, cv2.COLOR_BGR2GRAY)

    blur_step = cv2.GaussianBlur(grayscale_step, (5,5),0)

    edges_step = cv2.Canny(blur_step,50,150)



    kernel = np.ones((3,3), np.uint8)

    morph = cv2.morphologyEx(edges_step, cv2.MORPH_CLOSE,kernel)



    debug_mode = {

        "gray":grayscale_step,

        "blurred": blur_step,

        "edges" : edges_step,

        "morph" : morph

    }



    return morph, debug_mode