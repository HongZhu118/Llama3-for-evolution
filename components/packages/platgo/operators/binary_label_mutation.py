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


# promt = "Do you know what label coding is in an evolutionary algorithm,What is the crossover process of label coding"\
#         "What is the specific process of uniform crossover? Give an example"
# a = np.random.uniform(0,1,(1,10))
# b = np.random.uniform(0,1,(1,10))
a = np.random.randint(4, size=(3, 20))
b = np.random.randint(2, size=(3, 10))
p = 1/10
prompt = f"I have one existing numpy array a={a} .\
    Transform each bit with probability "+ str(p) +" into a random one of the elements in the row including itself.\
    Do not give additional explanations.\
    Give the new numpy array a directly"

s = get_response(prompt)

print(s)