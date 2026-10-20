from llama_index.core.readers import SimpleDirectoryReader
from llama_index.core import Settings, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai_like import OpenAILike
import llama_index
import os
from dotenv import load_dotenv

envs=load_dotenv()
def main(url:str)->None:
    # 1. 使用 DeepSeek 生成答案
    Settings.llm = OpenAILike(
        model="deepseek-flash",
        api_base="https://api.deepseek.com",
        api_key=os.environ['DeepSeek-API-KEY'],
        is_chat_model=True,
        context_window=1_000_000,
        max_tokens=2048,
        temperature=0.1,
        timeout=120,
    )

    # 2. 在本地生成向量，无需其他 API Key
    # 首次运行会下载模型；下载后在 CPU 上运行
    Settings.embed_model = HuggingFaceEmbedding(
        model_name="BAAI/bge-small-zh-v1.5",
        device="cpu",
    )
    document=SimpleDirectoryReader(url).load_data()
    index=VectorStoreIndex.from_documents(document)

    query_engine=index.as_query_engine()
    response=query_engine.query("泰勒是谁？")
    print(response)

if __name__ == '__main__':
    url="./data"
    main(url)
