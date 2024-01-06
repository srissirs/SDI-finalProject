import os
from openai import OpenAI

client = OpenAI(api_key='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')

chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": "How is Portugal?"
        }
    ],
    model="gpt-3.5-turbo",
)

print(chat_completion.choices[0].message.content)
