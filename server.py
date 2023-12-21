from fastapi import FastAPI, UploadFile, File
import time
import os

app = FastAPI()

@app.post('/upload')
async def upload_image(image: UploadFile = File(...)):
    file_location = f"./images/{image.filename}"
    txt_name = f"{image.filename.replace('.jpg', '')}.txt"
    txt_location = f"./images/{txt_name}"

    with open(file_location, 'wb') as file_object:
        file_object.write(image.file.read())

    while not os.listdir('./images').__contains__(txt_name):
        time.sleep(2)

    result = []
    with open(txt_location, 'r') as f:
        result.append([float(i) for i in f.readline().strip().replace('(','').replace(')','').split(',')])

    return { "data":result }