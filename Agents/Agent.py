import time
import logging
from google import genai
from google.genai import types
from dotenv import load_dotenv
import os
from google.genai.errors import ClientError

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('Agent')


class Agent:
    def __init__(self, system_prompt, top_p, top_k, temperature, response_mime_type, max_output_tokens=65535,
                 model="gemini-2.5-flash", timebuffer=3,create_chat=True):
        self.model = model
        self.config = types.GenerateContentConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            system_instruction=system_prompt,
            max_output_tokens=max_output_tokens,
            response_mime_type=response_mime_type,
            thinking_config=types.ThinkingConfig(thinking_budget=0)
        )
        load_dotenv()
        self.client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
        if create_chat:
            self.chat = self.client.chats.create(model=self.model, config=self.config)
        self.timebuffer = timebuffer

    def generate_chat_response(self, prompt):
        try:
            response = self.chat.send_message(prompt)
            return response
        except ClientError as e:
            print(e)
            logger.error(f"ClientError occurred: {str(e)}")
            error = e.details['error']['details']

            for detail in error:
                if detail.get("@type") == "type.googleapis.com/google.rpc.RetryInfo":
                    retry_str = detail.get("retryDelay")
                    retry_time = int(retry_str[:-1])
                    wait_time = retry_time + self.timebuffer

                    logger.warning(f"Rate limit exceeded. Waiting for {wait_time} seconds")
                    time.sleep(wait_time)

                    logger.info("Retrying request after rate limit wait")
                    return self.generate_chat_response(prompt)

    def generate_text_generation_response(self, prompt):
        try:
            response = self.client.models.generate_content(model=self.model,
                                                           config=self.config, contents=prompt)
            return response
        except ClientError as e:
            print(e)
            logger.error(f"ClientError occurred: {str(e)}")
            error = e.details['error']['details']

            for detail in error:
                if detail.get("@type") == "type.googleapis.com/google.rpc.RetryInfo":
                    retry_str = detail.get("retryDelay")
                    retry_time = int(retry_str[:-1])
                    wait_time = retry_time + self.timebuffer

                    logger.warning(f"Rate limit exceeded. Waiting for {wait_time} seconds")
                    time.sleep(wait_time)

                    logger.info("Retrying request after rate limit wait")
                    return self.generate_text_generation_response(prompt)

    def get_chat_history(self):

        chat_history = self.chat.get_history()
        print(chat_history)
        user_messages = []
        model_messages = []

        for message in chat_history:
            text_parts = []
            for part in message.parts:
                if hasattr(part, 'text') and part.text:
                    text_parts.append(part.text)

            full_text = ' '.join(text_parts)

            if full_text:
                if message.role == 'user':
                    user_messages.append(full_text)
                elif message.role == 'model':
                    model_messages.append(full_text)

        return {
            'user_messages': user_messages,
            'model_messages': model_messages
        }

    def count_token_price(self, response, input_price=0.30, output_price=2.5):
        print("Prompt tokens:", response.usage_metadata.prompt_token_count)
        print("Candidates tokens:", response.usage_metadata.candidates_token_count)
        print("Tool use tokens:", response.usage_metadata.tool_use_prompt_token_count)
        print("Thoughts tokens:", response.usage_metadata.thoughts_token_count)
        print("Total tokens:", response.usage_metadata.total_token_count)
        input_price = (response.usage_metadata.prompt_token_count / 1_000_000) * input_price
        output_price = (response.usage_metadata.candidates_token_count / 1_000_000) * output_price
        total_cost = input_price + output_price
        print(f"Cost: ${total_cost:.8f}")


