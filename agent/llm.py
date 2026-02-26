import os
import asyncio

from openai import OpenAI

DEEPSEEK_KEY = os.getenv("DEEPSEEK_API_KEY")
OPENAI_KEY   = os.getenv("OPENAI_API_KEY")
GEMINI_KEY   = os.getenv("GEMINI_API_KEY")
QWEN_KEY     = os.getenv("QWEN_API_KEY")
MOONSHOT_API_KEY = os.getenv("MOONSHOT_API_KEY")
XIAO_API_KEY = os.getenv("XIAO_API_KEY")
class LLM:
    def __init__(self,
                 server_name: str = "deepseek",
                 model: str = "deepseek-chat",
                 max_tokens: int = 4096,    
                 temperature: float = 0.7,
                 top_p: float = 1.0):
        if server_name == "deepseek":
            self.client = OpenAI(api_key=XIAO_API_KEY,
                                 #base_url="https://api.deepseek.com")
                                    base_url="https://aigc.x-see.cn/v1/",timeout=1000)
        elif server_name == "openai":
            self.client = OpenAI(api_key=XIAO_API_KEY
                                 #)
                                 ,base_url="https://aigc.x-see.cn/v1/",timeout=1000)
        elif server_name == "gemini":
            self.client = OpenAI(api_key=XIAO_API_KEY,
                                 base_url="https://aigc.x-see.cn/v1/",timeout=1000)
                                 #base_url="https://generativelanguage.googleapis.com/v1beta/openai")
        elif server_name == "qwen":
            self.client = OpenAI(api_key=QWEN_KEY,
                                 base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
        elif server_name == "kimi":
            self.client = OpenAI(api_key=MOONSHOT_API_KEY,
                                 base_url="https://api.moonshot.cn/v1")
        else:
            raise ValueError("server_name must be openai | deepseek | gemini")

        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p

    def chat(self, user: str, system: str = "You are a helpful CUDA optimization assistant.") -> str:
        while True:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user}
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                top_p=self.top_p
            )
            if resp.choices[0].message.content == "":
                continue
            return resp.choices[0].message.content
    
    def change_temperature(self, temperature: float) -> None:
        self.temperature = temperature

    def change_top_p(self, top_p: float) -> None:
        self.top_p = top_p

    async def achat(self,
                    user: str,
                    system: str = "You are a helpful CUDA optimization assistant.") -> str:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.chat, user, system)
    
if __name__ == "__main__":
    print(GEMINI_KEY)