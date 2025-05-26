# -*- coding: utf-8 -*-
import uvicorn
from PIL import Image
import numpy as np
import time
from fastapi import FastAPI, Request
import requests

from rag.index import describe_media
from respond import twilio_account_sid, twilio_auth_token, chat_with_phil, send_sms
import tensorflow as tf
from io import BytesIO
from twilio.twiml.messaging_response import MessagingResponse


# Keep the application running to listen for updates
# Check TensorFlow version
print(tf.__version__)  # This should print 2.15.0

#
# # 2. Create the app object
app = FastAPI()

# Medical image classification categories
MEDICAL_IMAGE_CATEGORIES = [
    'X-Ray_Normal',
    'X-Ray_Pneumonia',
    'X-Ray_COVID',
    'CT_Scan_Normal',
    'CT_Scan_Tumor',
    'MRI_Normal',
    'MRI_Brain_Tumor',
    'Ultrasound_Normal',
    'Ultrasound_Abnormal'
]

# Load medical image analysis models
XRAY_MODEL = tf.keras.models.load_model("models/xray_model.keras")
CT_SCAN_MODEL = tf.keras.models.load_model("models/ct_scan_model.keras")
MRI_MODEL = tf.keras.models.load_model("models/mri_model.keras")

@app.get('/')
def index():
    return {'message': 'MedAssist AI'}

@app.get('/{name}')
def get_name(name: str):
    return {'Welcome To MedAssist AI': f'{name}'}

def read_file_as_image(data) -> np.ndarray:
    # Use Pillow to convert the binary data to a NumPy array representing an image
    image = Image.open(BytesIO(data)).convert('RGB')

    # Resize the image to the expected input size of the model
    expected_size = (256, 256)  # Update with the actual input size expected by your model
    image = image.resize(expected_size)

    return np.array(image)


@app.post('/medical_analysis')
async def whatsapp_webhook(request: Request):
    form_data = await request.form()
    query = form_data.get("Body", "").strip().lower()
    image_url = form_data.get("MediaUrl0", "")
    user_phone_number = form_data.get('WaId', "")

    if query and not image_url:
        # send user question to llm
        return chat_with_phil(query, to=f"whatsapp:+{user_phone_number}")

    response = requests.get(image_url, auth=(twilio_account_sid, twilio_auth_token))

    if response.status_code != 200:
        return str(MessagingResponse().message("Failed to download the medical image. Please try again."))

    data = response.content

    input_image = read_file_as_image(data)
    img_batch = np.expand_dims(input_image, axis=0).astype(np.float32)

    if query == '':
        return chat_with_phil(describe_media(data), to=f"whatsapp:+{user_phone_number}", describe=True)

    elif query == "xray":
        predictions = XRAY_MODEL.predict(img_batch)
        predicted_class = MEDICAL_IMAGE_CATEGORIES[np.argmax(predictions[0])]
        confidence = float(np.max(predictions[0]))
        result = f"Analysis: {predicted_class}, Confidence: {confidence:.2f}"
        chat_with_phil(result, to=f"whatsapp:+{user_phone_number}")
    elif query == "ct":
        predictions = CT_SCAN_MODEL.predict(img_batch)
        predicted_class = MEDICAL_IMAGE_CATEGORIES[np.argmax(predictions[0])]
        confidence = float(np.max(predictions[0]))
        result = f"Analysis: {predicted_class}, Confidence: {confidence:.2f}"
        chat_with_phil(result, to=f"whatsapp:+{user_phone_number}")
    elif query == "mri":
        predictions = MRI_MODEL.predict(img_batch)
        predicted_class = MEDICAL_IMAGE_CATEGORIES[np.argmax(predictions[0])]
        confidence = float(np.max(predictions[0]))
        result = f"Analysis: {predicted_class}, Confidence: {confidence:.2f}"
        chat_with_phil(result, to=f"whatsapp:+{user_phone_number}")
    else:
        chat_with_phil(query, to=f"whatsapp:+{user_phone_number}")


if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8001)

#uvicorn app:app --reloadurn {"item_id": item_id, "q": q}
