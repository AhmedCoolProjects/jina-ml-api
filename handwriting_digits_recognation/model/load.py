import tensorflow as tf
from keras.models import model_from_json
import numpy as np


def init(image):
    graph = tf.compat.v1.get_default_graph()
    with graph.as_default():
        json_file = open(r'C:\Users\ahmed\Desktop\projects\backend\fast-api\jina-ml\handwriting_digits_recognation\model\model.json', "r")
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        loaded_model.load_weights(r'C:\Users\ahmed\Desktop\projects\backend\fast-api\jina-ml\handwriting_digits_recognation\model\model.h5')
        loaded_model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
        out = loaded_model.predict(image)
        print("------------",out)
        res = np.array_str(np.argmax(out, axis=1))
        return res
