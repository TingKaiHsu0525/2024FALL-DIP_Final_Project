import numpy as np
import matplotlib.pyplot as plt
from  CharacteristicModel import *
from ulit import *
from table1 import *
from CIECAM02 import *
from Inverse_CIECAM02 import *
import os


if __name__ == "__main__":
    #img = plt.imread("images/Bird.jpg")  # read image
    img = plt.imread("/home/pywu/Documents/DIP/DIP_term_project-main/DIP_term_project-main/code/images/Bird.jpg")

    size = img.shape

    # Device Characteristic Modeling
    XYZ = CharacteristicModel(img, M1, gamma1)
    XYZ = Normalized(XYZ)

    # Get white point
    img_white = np.ones_like(img)
    XYZw = CharacteristicModel(img_white, M1, gamma1)
    XYZw = Normalized(XYZw)
    
    # Device Characteristic Modeling
    XYZ1 = CharacteristicModel(img, M2, gamma2)
    XYZ1 = Normalized(XYZ1)

    # Get white point
    img_white = np.ones_like(img)
    XYZ1w = CharacteristicModel(img_white, M2, gamma2)
    XYZ1w = Normalized(XYZ1w)
    
    # CIECAM02
    High_h,High_J,High_C = CIECAM02(XYZ, XYZw).Forward()
    plt.figure()
    plt.imshow(High_h)
    plt.title("High_h")
    plt.figure()
    plt.imshow(High_J)
    plt.title("High_J")
    plt.figure()
    plt.imshow(High_C)
    plt.title("High_C")
    #Low_h,Low_J,Low_C = CIECAM02(XYZ1, XYZ1w).Forward()

    #print("Full display: ",High_h,High_J,High_C)
    #print("Low display: ",Low_h,Low_J,Low_C)

    XYZe = InverseCIECAM02(XYZ1, XYZ1w, High_h, High_J, High_C).Forward()
    print(XYZe.shape)
    print(XYZ.shape)



    #plt.imshow(img)
    #plt.title("Original img")
    #plt.figure()
    #plt.imshow(XYZ)
    #plt.title("XYZ")
    #$plt.figure()
    #plt.imshow(XYZw)
    #plt.title("XYZw")
    #plt.figure()
    #plt.imshow(XYZ1)
    #plt.title("XYZ1")
    #plt.figure()
    #plt.imshow(XYZ1w)
    #plt.title("XYZ1w")
    #plt.figure()
    #plt.imshow(XYZe)
    #plt.title("XYZe")
    plt.show()