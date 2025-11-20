from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
from datetime import datetime
import numpy as np
import pickle
import os
import pandas as pd
import traceback

app = FastAPI()

# -----------------------------
# CONFIG
# -----------------------------
MODEL_PATH = r"forecast.pkl"          # path to your trained model
DATA_CSV_PATH = r"meter_data.csv"     # same CSV you used for training


# -----------------------------
# Load model at startup
# -----------------------------
if os.path.exists(MODEL_PATH):
    try:
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        print("Model loaded successfully.")
    except Exception as e:
        print("Error loading model:", e)
        traceback.print_exc()
        model = None
else:
    print("WARNING: forecast.pkl not found.")
    model = None


# -----------------------------
# Build meter_id → int mapping
# -----------------------------
if os.path.exists(DATA_CSV_PATH):
    try:
        df_ids = pd.read_csv(DATA_CSV_PATH, usecols=["meter_id"])
        unique_ids = df_ids["meter_id"].unique().tolist()
        meter_map = {m: i for i, m in enumerate(unique_ids, start=1)}
        print("Meter map built:", meter_map)
    except Exception as e:
        print("Error while building meter_id map:", e)
        traceback.print_exc()
        meter_map = {}
else:
    print("WARNING: meter_data.csv not found, meter_id mapping empty.")
    meter_map = {}


# -----------------------------
# Helper HTML generators
# -----------------------------
def render_index_page(error_msg: str | None = None) -> str:
    error_html = f'<p style="color:red;">{error_msg}</p>' if error_msg else ""
    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>Meter Prediction</title>
    </head>
    <body>
        <h1>Meter Load Prediction</h1>
        {error_html}
        <form action="/predict" method="post">
            <label for="meter_id">Meter ID:</label>
            <input type="text" id="meter_id" name="meter_id" required>
            <br><br>

            <label for="date">Date:</label>
            <input type="date" id="date" name="date" required>
            <br><br>

            <label for="time">Time:</label>
            <input type="time" id="time" name="time" required>
            <br><br>

            <button type="submit">Predict</button>
        </form>
    </body>
    </html>
    """


def render_result_page(
    meter_id: str,
    date: str,
    time: str,
    prediction: float | None,
    error_msg: str | None = None,
) -> str:
    if error_msg:
        content_html = f'<p style="color:red;">{error_msg}</p>'
    else:
        content_html = f"""
        <p>Meter ID: {meter_id}</p>
        <p>Date: {date}</p>
        <p>Time: {time}</p>
        <p><strong>Predicted Load:</strong> {prediction}</p>
        """

    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>Prediction Result</title>
    </head>
    <body>
        <h1>Prediction Result</h1>
        {content_html}
        <br>
        <a href="/">Back</a>
    </body>
    </html>
    """


# -----------------------------
# Routes
# -----------------------------
@app.get("/", response_class=HTMLResponse)
async def index():
    html_content = render_index_page()
    return HTMLResponse(content=html_content)


@app.post("/predict", response_class=HTMLResponse)
async def predict(
    meter_id: str = Form(...),
    date: str = Form(...),
    time: str = Form(...),
):
    # 1. Parse date + time (support HH:MM and HH:MM:SS)
    try:
        try:
            dt = datetime.strptime(f"{date} {time}", "%Y-%m-%d %H:%M")
        except ValueError:
            dt = datetime.strptime(f"{date} {time}", "%Y-%m-%d %H:%M:%S")
    except ValueError as e:
        html_content = render_result_page(
            meter_id=meter_id,
            date=date,
            time=time,
            prediction=None,
            error_msg=f"Invalid date/time format: {e}",
        )
        return HTMLResponse(content=html_content, status_code=400)

    # 2. Encode meter_id
    meter_id_enc = meter_map.get(meter_id)
    if meter_id_enc is None:
        html_content = render_result_page(
            meter_id=meter_id,
            date=date,
            time=time,
            prediction=None,
            error_msg=f"Unknown meter_id '{meter_id}'. "
                      f"Make sure it exists in {DATA_CSV_PATH}.",
        )
        return HTMLResponse(content=html_content, status_code=400)

    # 3. Encode timestamp
    timestamp_enc = int(dt.timestamp())
    X = np.array([[meter_id_enc, timestamp_enc]])

    # 4. Predict
    if model is None:
        html_content = render_result_page(
            meter_id=meter_id,
            date=date,
            time=time,
            prediction=None,
            error_msg=f"Model not loaded. Check MODEL_PATH: {MODEL_PATH}",
        )
    else:
        try:
            print("X shape:", X.shape)
            if hasattr(model, "n_features_in_"):
                print("Model expects:", model.n_features_in_)
            pred = float(model.predict(X)[0])
            html_content = render_result_page(
                meter_id=meter_id,
                date=date,
                time=time,
                prediction=pred,
                error_msg=None,
            )
        except Exception as e:
            print("Prediction error:", e)
            traceback.print_exc()
            html_content = render_result_page(
                meter_id=meter_id,
                date=date,
                time=time,
                prediction=None,
                error_msg=f"Error during prediction: {e}",
            )

    return HTMLResponse(content=html_content)


# # Run with: uvicorn main:app --host 0.0.0.0 --port 9000 --reload
##############################################

# from fastapi import FastAPI, Form
# from fastapi.responses import HTMLResponse
# from datetime import datetime
# import numpy as np  # we'll use this for random number + rounding

# app = FastAPI()

# # -----------------------------
# # CONFIG: simulated load range
# # -----------------------------
# RANDOM_MIN = 8.9   # from your data
# RANDOM_MAX = 25.1  # from your data


# # -----------------------------
# # Helper HTML generators
# # -----------------------------
# def render_index_page(error_msg: str | None = None) -> str:
#     error_html = f'<p style="color:red;">{error_msg}</p>' if error_msg else ""
#     return f"""
#     <!DOCTYPE html>
#     <html>
#     <head>
#         <meta charset="UTF-8">
#         <title>Meter Prediction</title>
#     </head>
#     <body>
#         <h1>Meter Load Prediction </h1>
#         {error_html}
#         <form action="/predict" method="post">
#             <label for="meter_id">Meter ID:</label>
#             <input type="text" id="meter_id" name="meter_id" required>
#             <br><br>

#             <label for="date">Date:</label>
#             <input type="date" id="date" name="date" required>
#             <br><br>

#             <label for="time">Time:</label>
#             <input type="time" id="time" name="time" required>
#             <br><br>

#             <button type="submit">Predict</button>
#         </form>
#     </body>
#     </html>
#     """


# def render_result_page(
#     meter_id: str,
#     date: str,
#     time: str,
#     prediction: float | None,
#     error_msg: str | None = None,
# ) -> str:
#     if error_msg:
#         content_html = f'<p style="color:red;">{error_msg}</p>'
#     else:
#         content_html = f"""
#         <p>Meter ID: {meter_id}</p>
#         <p>Date: {date}</p>
#         <p>Time: {time}</p>
#         <p><strong>Predicted Load :</strong> {prediction}</p>
#         """

#     return f"""
#     <!DOCTYPE html>
#     <html>
#     <head>
#         <meta charset="UTF-8">
#         <title>Prediction Result</title>
#     </head>
#     <body>
#         <h1>Prediction Result</h1>
#         {content_html}
#         <br>
#         <a href="/">Back</a>
#     </body>
#     </html>
#     """


# # -----------------------------
# # Routes
# # -----------------------------
# @app.get("/", response_class=HTMLResponse)
# async def index():
#     html_content = render_index_page()
#     return HTMLResponse(content=html_content)


# @app.post("/predict", response_class=HTMLResponse)
# async def predict(
#     meter_id: str = Form(...),
#     date: str = Form(...),
#     time: str = Form(...),
# ):
#     # 1. Parse date + time (support HH:MM and HH:MM:SS)
#     try:
#         try:
#             dt = datetime.strptime(f"{date} {time}", "%Y-%m-%d %H:%M")
#         except ValueError:
#             dt = datetime.strptime(f"{date} {time}", "%Y-%m-%d %H:%M:%S")
#     except ValueError as e:
#         html_content = render_result_page(
#             meter_id=meter_id,
#             date=date,
#             time=time,
#             prediction=None,
#             error_msg=f"Invalid date/time format: {e}",
#         )
#         return HTMLResponse(content=html_content, status_code=400)

#     # 2. SIMULATED prediction: random value in [8.9, 25.1], 3 decimal places
#     simulated_value = float(np.round(np.random.uniform(RANDOM_MIN, RANDOM_MAX), 3))

#     html_content = render_result_page(
#         meter_id=meter_id,
#         date=date,
#         time=time,
#         prediction=simulated_value,
#         error_msg=None,
#     )
#     return HTMLResponse(content=html_content)


# # Run with: uvicorn main:app --host 0.0.0.0 --port 9000 --reload
