import openai

openai.api_key = "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"
messages = []
print("您好，我们终于见面了！！您希望接下来我为您提供什么服务？")
system_message = input("人类说：")
messages.append({"role":"system","content":system_message})
print("\n")

print("好的，明白了! 我会服务好您的。" + "\n" + "现在请和我聊天吧！" + "\n" + "记住，烦我的时候，请说‘再见’")

while True:
    # Collect the user's message
    print("\n")
    message = input("人类说：")
    messages.append({"role":"user","content": message})

    response=openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    # model="gpt-4",
    messages=messages
    )

    reply = response["choices"][0]["message"]["content"]
    print("ChatGPT说: ", reply)

    # Check if the user wants to exit the conversation
    if message.lower() == "再见":
        break