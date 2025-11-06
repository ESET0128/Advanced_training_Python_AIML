from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
 
# Load the iris dataset
iris = load_iris()
X, y = iris.data, iris.target
 
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
 
# Train a RandomForest classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
 
# Evaluate the model
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=iris.target_names)
 
print(f"Model Accuracy: {accuracy}")
print("Classification Report:")
print(report)
 
# Save the trained model to a file
joblib.dump(clf, "iris_model.pkl")
 
 
# from fastapi import FastAPI, Request, Form
# # from fastapi.responses import HTMLResponse
# # from fastapi.templating import Jinja2Templates
# from pydantic import BaseModel
# import joblib
# import numpy as np
# from sklearn.datasets import load_iris
 
# # Load the trained model
# model = joblib.load("iris_model.pkl")
 
# app = FastAPI()
 
# class IrisInput(BaseModel):
#     sepal_length: float
#     sepal_width: float
#     petal_length: float
#     petal_width: float
 
# class IrisPrediction(BaseModel):
#     predicted_class: int
#     predicted_class_name: str
 
 
# @app.get("/")
# def read_root():
#     return {"Hello": "World"}
 
# @app.post("/predict", response_model=IrisPrediction)
# def predict(data: IrisInput):
 
#     # prepare input data
#     input_data = np.array([[data.sepal_length, data.sepal_width, data.petal_length, data.petal_width]])
 
#     # make prediction
#     predicted_class = model.predict(input_data)[0]
#     predicted_class_name = load_iris().target_names[predicted_class]
#     return IrisPrediction(predicted_class=predicted_class, predicted_class_name=predicted_class_name)
 
# if _name_ == "_main_":
#     import uvicorn
#     uvicorn.run(app, host="127.0.0.1", port=8000)
 


 
# from fastapi import FastAPI, Request, Form
# from fastapi.responses import HTMLResponse
# from fastapi.templating import Jinja2Templates
# from pydantic import BaseModel
# import joblib
# import numpy as np
# from sklearn.datasets import load_iris
 
# # Load the trained model
# model = joblib.load("iris_model.pkl")
 
# app = FastAPI()
 
# # Set up templates directory
# templates = Jinja2Templates(directory="template")
 
 
# class IrisInput(BaseModel):
#     sepal_length: float
#     sepal_width: float
#     petal_length: float
#     petal_width: float
 
# class IrisPrediction(BaseModel):
#     predicted_class: int
#     predicted_class_name: str
 
 
# @app.get("/", response_class=HTMLResponse)
# async def read_root(request: Request):
#     return templates.TemplateResponse("index.html", {"request": request})
   
 
 
# @app.post("/predict", response_model=IrisPrediction)
# def predict(request: Request,
#             sepal_length: float = Form(...),
#             sepal_width: float = Form(...),
#             petal_length: float = Form(...),
#             petal_width: float = Form(...)):
#     # prepare input data
#     input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
 
#     # make prediction
#     predicted_class = model.predict(input_data)[0]
#     predicted_class_name = load_iris().target_names[predicted_class]
#     return templates.TemplateResponse(
#         "result.html",
#         {"request": request,
#          "predicted_class": predicted_class,
#          "predicted_class_name": predicted_class_name,
#          "sapal_length": sepal_length,
#          "sapal_width": sepal_width,
#          "petal_length": petal_length,
#          "petal_width": petal_width})
# if _name_ == "_main_":
#     import uvicorn
#     uvicorn.run(app, host="127.0.0.1", port=8000)