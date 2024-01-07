import os
from openai import OpenAI

client = OpenAI(api_key='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')

chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": "Hi, how are you?"
        }
    ],
    model="gpt-3.5-turbo",
)

print(chat_completion.choices[0].message.content)
