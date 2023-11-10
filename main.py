import pandas as pd
from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
import uvicorn
from starlette.staticfiles import StaticFiles
from datetime import datetime

from model import log_predict
from svm import svm_predict
from kmeans import MiniBatchKMeans_predict


app = FastAPI()
templates = Jinja2Templates(directory="templates")
port = 10000
app.mount("/static", StaticFiles(directory="static", html=True), name="static")


@app.get("/")
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/get_loan")
async def root(request: Request):
    months = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
    m_idx = int(datetime.now().strftime("%m"))
    day = int(datetime.now().strftime("%d"))
    return templates.TemplateResponse("loan.html", {"request": request, "month": months[m_idx - 1], "day": day})


from pydantic import BaseModel


class Item(BaseModel):
    age: int
    job: str
    marital: str
    education: str
    default: str
    balance: float
    housing: str
    loan: str
    contact: str
    day: int
    month: str
    duration: int
    campaign: int
    pdays: int
    previous: int


@app.post("/get_score")
async def root(item: Item):
    prediction = log_predict(item.age, item.job, item.marital, item.education, item.default, item.balance, item.housing,
                             item.loan,
                             item.contact, item.day, item.month, item.duration, item.campaign)

    prediction_list = prediction.tolist()
    print(prediction_list)
    return prediction_list


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=port)
    print(f"Running on localhost: {port}")
