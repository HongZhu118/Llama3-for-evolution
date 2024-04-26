# Llama3-for-evolution

## 1. 下载ollama
[下载地址](https://ollama.com/)

## 2. 下载模型
llama3-8B下载示例：
```
ollama run llama3
```
[官方文档](https://github.com/ollama/ollama)

## 3. 安装依赖包
```
pip3 install -r requirements.txt
```
## 4. 使用ollama提供的本地接口调用llama3 
```
curl http://localhost:11434/api/chat -d '{
"model": "llama3",
 "messages": [
 { "role": "user", "content": "why is the sky blue?" }
 ]
}'
```
