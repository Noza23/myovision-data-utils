import nest_asyncio
from pyngrok import ngrok
import uvicorn


with open(".env", "r") as f:
    for line in f.readlines():
        if "ngrok_api" in line:
            api_key = line.split("=")[1].split('"')[1]

ngrok_tunnel = ngrok.connect(8000)
ngrok.set_auth_token(api_key)
print('Public URL:', ngrok_tunnel.public_url)
nest_asyncio.apply()
uvicorn.run("mask_generator:app", port=8000)