import openai
from openai import OpenAI
import logging
from dotenv import load_dotenv
import os
import json
import requests
from PIL import Image
import numpy as np
import random
import gradio as gr
import time

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables from .env file
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    logging.error("OpenAI API key not found in .env file.")
    exit(1)

# Initialize OpenAI client
client = OpenAI(api_key=openai_api_key)

# Function to generate image using DALL-E
def generate_dalle_image(prompt):
    response = client.images.generate(
        model="dall-e-3",
        prompt=prompt,
        size="1792x1024",
        quality="hd",
        n=1,
    )
    return response.data[0].url

# Define tools for image generation
tools = [
    {
        "type": "function",
        "function": {
            "name": "generate_dalle_image",
            "description": "Generates an image based on a user prompt using DALL-E.",
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "A text prompt to generate an image from."
                    },
                },
                "required": ["prompt"]
            },
        }
    }
]

# Global variable to store messages
messages = []

# Function to get chatbot response
def bot_response(user_prompt, max_retries=3):
    global messages

    messages.append({"role": "user", "content": user_prompt})

    attempt = 0
    while attempt < max_retries:
        try:
            completion = client.chat.completions.create(
                model="gpt-4-1106-preview",
                messages=messages,
                tools=tools,
                tool_choice="auto"
            )
            response = completion.choices[0].message
            print(response)
            tool_call_made = False

            if response.tool_calls:
                for tool_call in response.tool_calls:
                    if tool_call.function.name == "generate_dalle_image":
                        print(response)
                        tool_call_made = True
                        try:
                            prompt = json.loads(tool_call.function.arguments)["prompt"]
                        except json.JSONDecodeError as e:
                            logging.error(f"JSON decoding error: {e}")
                            logging.error(f"Faulty JSON string: {tool_call.function.arguments}")
                            return "An error occurred while processing the response."

                        image_url = generate_dalle_image(prompt)
                        image = Image.open(requests.get(image_url, stream=True).raw)
                        filename = image_url.split("/")[-1].split("?")[0] + str(np.random.randint(1000)) + ".png"
                        image.save(filename)
                        image.show()
                        messages.append({"role": "assistant", "content": f"Image URL: {image_url}"})
                        return f"Image URL: {image_url}"

            if not tool_call_made:
                messages.append({"role": "assistant", "content": response.content})

            if len(messages) > 12:
                messages = messages[-12:]

            return response.content

        except Exception as e:
            logging.error(f"An error occurred in bot_response: {e}")
            if 'content_policy_violation' in str(e):
                logging.info("Retrying due to content policy violation...")
                attempt += 1
                time.sleep(1)
            else:
                return "An error occurred while generating the response."

    return "Failed to generate response after retries."
            
        
def main():
    while True:
        user_input = input("Enter your prompt (or 'exit' to quit): ")
        if user_input.lower() == 'exit':
            break

        response = bot_response(user_input)
        print(f"Response: {response}")


with gr.Blocks() as demo:
    chatbot = gr.Chatbot()
    msg = gr.Textbox()
    clear = gr.ClearButton([msg, chatbot])

    def respond(message, chat_history):
        # Use your existing bot_response function
        bot_message = bot_response(message)
        
        # Append to chat history
        chat_history.append((message, bot_message))
        time.sleep(2)  # Optional delay for more natural conversation flow

        return "", chat_history

    msg.submit(respond, [msg, chatbot], [msg, chatbot])

if __name__ == "__main__":
    demo.launch()
