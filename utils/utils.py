import os
from dotenv import load_dotenv
from twilio.rest import Client
from langchain_core.tools import ToolException
load_dotenv()



def _handle_error(error: ToolException) -> str:
    return (
        "The following errors occurred during tool execution:"
        + error.args[0]
        + "Please try another tool."
    )

def format_response(response):
    # Extract the output text from the response
    output_text = response.get('output', '')

    # Replace tab characters with two spaces for better readability
    formatted_text = output_text.replace('\t', '  ')

    # Replace newline characters with actual newlines
    formatted_text = formatted_text.replace('\\n', '\n')

    return formatted_text

def format_docs(docs):
    return  "\n\n---\n\n".join([doc.page_content for doc in docs])


