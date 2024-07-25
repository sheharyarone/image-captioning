from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
import os
from random import randint
import uuid
from model import image_caption
from imutils import paths
import argparse
import pandas as pd
import time
import sys
import cv2
import os

IMAGEDIR = "images/"

app = FastAPI()

captions={}
df = pd.DataFrame(columns=['id', 'hash', 'caption'])



@app.get('/{id}')
async def func(id:int):
    global df
    if id in df['id'].values:
        row = df[df['id'] == id]
        return {"RESPONSE FROM OUR MODEL :":row["caption"]}
    return {"reponse": "Hello World! NOT FOUND ID"}


@app.post("/upload/")
async def create_upload_file(file: UploadFile = File(...)):
    global df
    file.filename = f"{uuid.uuid4()}.jpg"
    contents = await file.read()

    #save the 
    image_direct = IMAGEDIR + file.filename
    with open(f"{IMAGEDIR}{file.filename}", "wb") as f:
        f.write(contents)
    image = cv2.imread(image_direct)
    if image is None:
        return ("ISSUE")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(image, (8 + 1, 8))
    diff = resized[:, 1:] > resized[:, :-1]
    imageHash = sum([2 ** i for (i, v) in enumerate(diff.flatten()) if v])

    if len(df)!=0 and imageHash in df['hash'].values :
        row = df[df['hash'] == imageHash]

        return {"RESPONSE FROM OUR MODEL :":row["caption"]}

    file.filename = f"{uuid.uuid4()}.jpg"
    contents = await file.read()

    #need to check below code whether it is necessary or not 
    with open(f"{IMAGEDIR}{file.filename}", "wb") as f:
        f.write(contents)
    response = await image_caption(image_direct)
    #SAVING THE RESPONSE IN OUR DICTIONARY 
    entry = {'id': len(df), 'hash': imageHash, 'caption': response}
    df.loc[len(df)] = entry     
    return {"Response from our model": response}


