# pip install zhipuai 

from zhipuai import ZhipuAI

client = ZhipuAI(api_key="a2039bd00af172198d628d93a0ff9769.QgESVfAiyj13e5Vm")

response = client.chat.completions.create(
    model="glm-4",  # model name
    messages=[
        {
            "role": "user",
            "content": "write a excited fairy story. Limited as 2500 chars."
        }
    ],
)

print(response.choices[0].message.content)
# for trunk in response:
#     print(trunk)