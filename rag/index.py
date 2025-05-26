import json
import os
from dotenv import load_dotenv
import io
from PIL import Image
from langsmith import traceable
import requests
import time
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI, HarmBlockThreshold, \
    HarmCategory
import google.generativeai as genai
from exa_py import Exa
from langchain.agents import tool, AgentExecutor, create_react_agent
from langchain.tools.retriever import create_retriever_tool

from rag.prompt import prompt
from utils.utils import format_response
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

import json

load_dotenv()

# Initialize Firebase Admin SDK
# cred_obj = credentials.Certificate('./rag/config.json')

# default_app = firebase_admin.initialize_app(cred_obj, {
#     'databaseURL': 'https://rusero-2cbc0-default-rtdb.europe-west1.firebasedatabase.app',
#     'databaseAuthVariableOverride': None
# })
# client = firestore.client(app=default_app)

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
exa = Exa(api_key=os.environ["EXA_KEY"])

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
new_db = FAISS.load_local("./rag/faiss_index", embeddings,
                          allow_dangerous_deserialization=True)


llm_model = ChatGoogleGenerativeAI(model="models/gemini-2.0-flash", temperature=0, safety_settings={
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
})

# doc retriever tool
retriever = new_db.as_retriever(search_kwargs={'k': 1})
retriever_tool = create_retriever_tool(
    retriever,
    "retriever",
    "Query a retriever to get information about medical conditions, symptoms, treatments, and healthcare guidelines. "
    "Help analyze medical symptoms and provide evidence-based recommendations for patient care."
)





@tool
def search(query: str):
    """Search for a webpage based on the query."""
    return exa.search_and_contents(f"{query}", use_autoprompt=True, num_results=5, type="auto", text=True)


# class AddReminderSchema(BaseModel):
#     phone_number: str
#     message: str
#     remind_at: datetime

# def get_patient_data(path) -> dict:
#     # Get the data from the specified path
#     snapshot = db.reference(path).get()

#     # Create a new dictionary to store the data
#     patient_data = {}

#     # Iterate through the snapshot items and add them to the patient_data dictionary
#     for key, val in snapshot.items():
#         farm_data[key] = val

#     # Return the farm_data dictionary
#     return patient_data



def read_file_as_image(data):
    # If data is a file path, open and read the image directly
    if isinstance(data, str):
        return Image.open(data)
    # If data is binary data, use io.BytesIO to read it
    elif isinstance(data, bytes):
        return Image.open(io.BytesIO(data))
    else:
        raise ValueError("Unsupported data format for image reading")


def describe_media(data):
    """Analyzes medical images supplied by users for medical diagnosis support"""
    system_instruction = (
        "You are a medical assistant called MedAssist. A user has uploaded a medical image. Your task is to "
        "analyze the image and provide preliminary medical insights. Identify any visible medical conditions, "
        "abnormalities, or areas of concern. Provide detailed context about the medical image while maintaining "
        "professional medical terminology. If you notice any concerning features, explain them clearly and suggest "
        "next steps for medical evaluation. Remember to emphasize that this is a preliminary analysis and should "
        "be followed up with professional medical consultation. Summarize responses to 1600 characters or less.")

    print("Making LLM inference request...")

    # Read the image from the input data
    input_image = read_file_as_image(data)

    # Save the image to a temporary file for uploading
    temp_image_path = "temp_image.jpg"
    input_image.save(temp_image_path)

    media_file = genai.upload_file(path=temp_image_path)
    print(f"Completed upload: {media_file.uri}")

    while media_file.state.name == "PROCESSING":
        print('Waiting for media to be processed.')
        time.sleep(10)
        media_file = genai.get_file(media_file.name)

    if media_file.state.name == "FAILED":
        raise ValueError(media_file.state.name)
    print(f'media processing complete: ' + media_file.uri)

    # Set the model to Gemini 1.5 Flash.
    model = genai.GenerativeModel(model_name="gemini-1.5-pro-latest")

    response = model.generate_content([system_instruction, media_file],
                                      request_options={"timeout": 600})

    print(response.text)
    return response.text


#API CALL TO GET WEATHER DATA OR ANY DATA OF CHOICE
@tool
def fetch_weather(city):
    """
    Fetches weather data from  API to get city weather information.
    """

    # Base URL with placeholders for API key, latitude, and longitude
    url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&APPID={os.getenv('WEATHER_KEY')}&units=metric"

    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for unsuccessful requests
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching weather data: {e}")
        return None


# setup tools and agent
message_history = ChatMessageHistory()

tools = [retriever_tool, search, fetch_weather]


@traceable
def user_input(user_question):


    agent = create_react_agent(llm_model, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

    # active_farmer = get_user_by_phone_number(to.replace("whatsapp:", ""))


    agent_with_chat_history = RunnableWithMessageHistory(
        agent_executor,
        lambda session_id: message_history,
        input_messages_key="input",
        history_messages_key="chat_history",
    )

    response = agent_with_chat_history.invoke(
        {
            "input": f"{user_question}"
            }

        ,
        # This is needed because in most real world scenarios, a session id is needed
        # It isn't really used here because we are using a simple in memory ChatMessageHistory
        config={"configurable": {"session_id": 1234567890}},
    )
    print(response)
    formatted_response = format_response(response)
    return formatted_response
