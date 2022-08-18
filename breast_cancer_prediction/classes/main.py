import os
import pickle as pkl



class BreastCancerPrediction():
    def __init__(self, data_list):
        model_path_pardir = os.path.join(os.path.dirname(__file__), os.path.pardir)
        model_path = os.path.join(model_path_pardir, "utils", "lr_model.pkl")
        sc_path = os.path.join(model_path_pardir, "utils", "sc.pkl")
        self.data_list = data_list
        with open(model_path, 'rb') as f:
            self.model = pkl.load(f)
        with open(sc_path, 'rb') as f:
            self.sc = pkl.load(f)
        # self.data = [[ 1.06460408 , 0.94657216 , 0.95251147, -1.02909365, -0.64172985 ,-0.02136582]]
    def predict(self):
        data = self.sc.transform(self.data_list)
        predicted_value = self.model.predict(data).tolist()
        accuracy = self.model.predict_proba(data).tolist()
        return {
            "predicted_value": predicted_value,
            "proba_element": accuracy[0][predicted_value[0]],
            "scaled_list": data.tolist(),
            "predicted_result": "Benign" if predicted_value[0] == 0 else "Malignant"
        }