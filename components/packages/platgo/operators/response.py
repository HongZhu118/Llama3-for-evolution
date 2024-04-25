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


a = np.random.randint(2, size=(1, 20))
b = np.random.randint(2, size=(1, 20))
print("a:", a)
print("b:", b)
prompt = (f"""There are two numpy array here:
             numpy array a:<{a}> 
           numpy array b:<{b}>
            Your task is to return a numpy array c based on a and b that satisfies the following conditions:\
          1- numpy array c s the same size as numpy array a\
          2- The elements of numpy array c are obtained by randomly swapping the corresponding positions of numpy array a and b\
          
         
          Use the following format:\
          a:<numpy array a>\
          b:<numpy array b>\
          c:<numpy array c>\
               """)

s = get_response(prompt)
# 第一步：替换字符串中不需要的字符并转换格式
# 去掉换行符，并在数字之间添加逗号以符合Python列表的格式
formatted_str = s.replace('\n', ',').replace(' ', ', ')

# 由于上面的替换可能导致连续的逗号出现，我们需要进一步处理这个字符串
# 移除连续的逗号
while ',,' in formatted_str:
    formatted_str = formatted_str.replace(',,', ',')

# 第二步：将格式化后的字符串转换为Python列表
# 使用eval函数将字符串转换为实际的Python对象（在这里是一个二维列表）
array_list = eval(formatted_str)

# 第三步：将这个列表转换为numpy数组
array_np = np.array(array_list)

# 打印结果
print(array_np)


print(s)