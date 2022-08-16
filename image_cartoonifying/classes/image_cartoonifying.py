import cv2  # for image processing
import numpy as np  # to store image


class Cartoonifying:
    def __init__(self, uploaded_file):
        self.image = cv2.imdecode(np.frombuffer(uploaded_file.file.read(), np.uint8), 1)
        self.image_gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        res, self.image_result = cv2.imencode(".png", self.image_gray)
        self.smooth_gray = cv2.medianBlur(self.image_gray, 5)
    def get_edges(self):
        edges =  cv2.adaptiveThreshold(
            self.smooth_gray,
            255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY,
            9,
            9
        )
        return edges
    def coloring(self):
        self.coloring_image1 = cv2.bilateralFilter(self.image, 9, 300, 300)
        self.coloring_image = cv2.bitwise_and(self.coloring_image1, self.coloring_image1, mask=self.get_edges())
        res, self.image_result = cv2.imencode(".png", self.coloring_image)
        return self.image_result
    def cartoonify(self):
        return self.coloring()