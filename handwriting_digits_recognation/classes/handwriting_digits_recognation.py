import base64
import re
from cv2 import imread, resize
import numpy as np
import sys
import os
from handwriting_digits_recognation.model.load import init

class HandwritingDigitsRecognation():
    
    def __init__(self, imageDataURL):
        self.imageDataURL = imageDataURL
        self.imageDataURL = self.imageDataURL.encode('utf-8')

    def convertImage(self):
        imgStr = re.search(b'base64,(.*)', self.imageDataURL).group(1)
        with open("output.png", "wb") as output:
            output.write(base64.decodebytes(imgStr))
    
    def prepare_image(self):
        self.convertImage()
        self.image = imread("output.png")
        self.image = np.invert(self.image)
        self.image = self.image[:, :, 0]
        self.image = resize(self.image, (28, 28))
        self.image = self.image.reshape((1, 28, 28, 1))
    
    def predict(self):
        self.prepare_image()
        return init(self.image)



