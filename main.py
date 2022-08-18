import datetime
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
# from code_esi_chatbot.classes.ChatBot import ChatBot
# from handwriting_digits_recognation.classes.handwriting_digits_recognation import HandwritingDigitsRecognation
from image_cartoonifying.classes.image_cartoonifying import Cartoonifying
from breast_cancer_prediction.classes.main import BreastCancerPrediction




app = FastAPI()

origins = ["http://localhost:5173","http://127.0.0.1:5173","https://jina-ml.vercel.app","https://ml.ahmedbargady.me"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET","POST"],
    allow_headers=["*"]
)


@app.get("/")
def welcome():
    return {"message": "Welcome to Jina ML API"}


# --------------- function that will send us the images path --------------
## put the filename in the endpoint: ?filename=...

@app.get("/api/ml/files")
def get_file(filename: str):
    file_location = f"files/upload/{filename}"
    return FileResponse(file_location)

# ---------------- https://jina-ml.vercel.app/project/image-cartoonify ---------------------

@app.get("/api/ml/image-cartoonify")
def image_cartoonify():
    return {"message": "Image Cartoonify"}

@app.post("/api/ml/image-cartoonify")
async def image_cartoonify_post(uploaded_image: UploadFile = File(...)):
    file_name = f"{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}-{uploaded_image.filename}"
    file_location = f"files/upload/{file_name}"
    # start model
    cartoonifyingObject = Cartoonifying(uploaded_image)
    image_result = cartoonifyingObject.cartoonify()
    # end model
    with open(file_location, "wb+") as file_object:
        file_object.write(image_result)
    return file_name

# ---------------- https://jina-ml.vercel.app/project/breast-cancer-prediction ---------------------

class breastCancerPredictionPost(BaseModel):
    data_list: list

@app.post("/api/ml/breast-cancer-prediction")
async def breast_cancer_prediction(data: breastCancerPredictionPost):
    # start model
    breastCancerPredictionObject = BreastCancerPrediction(data.data_list)
    result = breastCancerPredictionObject.predict()
    # end model
    return {"result":result}

# # for handwriting digits recognation

# class imageDataURLClass(BaseModel):
#     imageDataURL: str

# @app.get("/api/ml/handwriting_digits_recognation")
# def handwriting_digits_recognation():
#     return {"message": "Handwriting Digits Recognation"}

# @app.post("/api/ml/handwriting_digits_recognation")
# async def handwriting_digits_recognation_post(data: imageDataURLClass):
#     # model
#     HandwritingDigitsRecognationObject = HandwritingDigitsRecognation(data.imageDataURL)
#     result = HandwritingDigitsRecognationObject.predict()
#     # end model
#     return result
# bot = ChatBot()
# @app.post("/api/ml/code_esi_chatbot")
# async def code_esi_chatbot(user_input: str):
#     # for the terminal test
#     return bot.get_result(user_input)

# @app.get("/api/ml/code_esi_chatbot")
# def code_esi_chatbot_get():
#     return {"message": "code_esi_chatbot"}
