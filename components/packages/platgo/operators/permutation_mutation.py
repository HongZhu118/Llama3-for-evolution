import json
import http
import http.client
import numpy as np
def get_response(prompt_content):
    payload_explanation = json.dumps({
        "model": "gpt-3.5-turbo-1106",
        # "model": "gpt-4-0613",
        "messages": [
            {
                "role": "user",
                "content": prompt_content
            }
        ],
        "safe_mode": False
    })
    headers = {
        'Authorization': 'Bearer ' + "fk222825-oRE9Oqi0Cp6MvPqu1s6tROFE9K3AkqFJ",
        'User-Agent': 'Apifox/1.0.0 (https://apifox.com)',
        'Content-Type': 'application/json',
        'x-api2d-no-cache': 1
    }

    while True:
        try:
            conn = http.client.HTTPSConnection("oa.api2d.net")
            conn.request("POST", "/v1/chat/completions", payload_explanation, headers)
            res = conn.getresponse()
            data = res.read()
            json_data = json.loads(data)
            response = json_data['choices'][0]['message']['content']
            break
        except:
            print("Error in API. Restarting the process...")
            continue
    return response

N = 10
# 生成一个随机序列编码
a = np.zeros([3,10],dtype=int)
for i in range(len(a)):
       random_sequence = np.random.permutation(N) + 1  # 加1是因为numpy的permutation从0开始
       a[i] = random_sequence
print(a)

prompt = f"I have one existing numpy array a={a} .\
    Please help me swap the positions of two random elements in each row \
    Do not give additional explanations.\
    Give the new numpy array a directly"
s = get_response(prompt)
print(s)