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

CLASS_NAMES = ['Tomato_Bacterial_spot',
               'Tomato_Early_blight',
               'Tomato_Late_blight',
               'Tomato_Leaf_Mold',
               'Tomato_Septoria_leaf_spot',
               'Tomato_Spider_mites_Two_spotted_spider_mite',
               'Tomato__Target_Spot',
               'Tomato__Tomato_YellowLeaf__Curl_Virus',
               'Tomato__Tomato_mosaic_virus',
               'Tomato_healthy']
corn_CLASS_NAMES = ['Corn Blight', 'Corn Common_Rust', 'Corn Gray_Leaf_Spot', 'Healthy Corn']
potato_CLASS_NAMES = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']

MODEL = tf.keras.models.load_model("models/tomato_model_v2.keras")
potato_MODEL = tf.keras.models.load_model("models/potato.keras")
corn_MODEL = tf.keras.models.load_model("models/corn_model_v1.keras")

@app.get('/')
def index():
    return {'message': 'RUSERO AI'}

@app.get('/{name}')
def get_name(name: str):
    return {'Welcome To RuseroAi': f'{name}'}

def read_file_as_image(data) -> np.ndarray:
    # Use Pillow to convert the binary data to a NumPy array representing an image
    image = Image.open(BytesIO(data)).convert('RGB')

    # Resize the image to the expected input size of the model
    expected_size = (256, 256)  # Update with the actual input size expected by your model
    image = image.resize(expected_size)

    return np.array(image)


@app.post('/rusero')
async def whatsapp_webhook(request: Request):

    form_data = await request.form()
    isifo = form_data.get("Body", "").strip().lower()
    image_url = form_data.get("MediaUrl0", "")
    user_phone_number = form_data.get('WaId', "")
    # if not image_url:
    #     return chat_with_phil(str("Please send an image of the isifo for prediction."))
    print(form_data)

    if isifo and not image_url:
        # send user question to llm
        return chat_with_phil(isifo, to=f"whatsapp:+{user_phone_number}")

    response = requests.get(image_url, auth=(twilio_account_sid, twilio_auth_token))

    if response.status_code != 200:
        return str(MessagingResponse().message("Failed to download the image. Please try again."))

    data = response.content

    input_image = read_file_as_image(data)
    img_batch = np.expand_dims(input_image, axis=0).astype(np.float32)

    if isifo == '':
        return chat_with_phil(describe_media(data), to=f"whatsapp:+{user_phone_number}", describe=True)

    elif isifo == "potato":
        predictions = potato_MODEL.predict(img_batch)
        predicted_class = potato_CLASS_NAMES[np.argmax(predictions[0])]
        confidence = float(np.max(predictions[0]))
        result = f"class, {predicted_class}, Confidence, {confidence}"
        chat_with_phil(result, to=f"whatsapp:+{user_phone_number}")
    elif isifo == "corn":
        predictions = corn_MODEL.predict(img_batch)
        predicted_class = corn_CLASS_NAMES[np.argmax(predictions[0])]
        confidence = float(np.max(predictions[0]))
        result = f"class, {predicted_class}, Confidence, {confidence}"
        chat_with_phil(result, to=f"whatsapp:+{user_phone_number}")
    elif isifo == "tomato":
        predictions = MODEL.predict(img_batch)
        predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
        confidence = float(np.max(predictions[0]))
        result = f"class, {predicted_class}, Confidence, {confidence}"
        chat_with_phil(result, to=f"whatsapp:+{user_phone_number}")
    else:
        chat_with_phil(isifo, to=f"whatsapp:+{user_phone_number}")


if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8001)

#uvicorn app:app --reloadurn {"item_id": item_id, "q": q}

