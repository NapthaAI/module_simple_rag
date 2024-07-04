import os
from typing import Dict
from simple_rag.schemas import InputSchema
from naptha_sdk.utils import get_logger
import chromadb
import chromadb.utils.embedding_functions as embedding_functions
from litellm import completion
from dotenv import load_dotenv

load_dotenv()
logger = get_logger(__name__)


def get_retrieved_docs_str(retrieved_docs):
    docs = retrieved_docs['documents'][0]
    return '\n'.join([f'Doc {i}:\n{doc}\n\n' for i, doc in enumerate(docs)])

def run(inputs: InputSchema, worker_nodes = None, orchestrator_node = None, flow_run = None, cfg: Dict = None):
    logger.info(f"Running module with prompt: {inputs.question}")
    logger.info(f"Input directory: {inputs.input_dir}")

    client = chromadb.PersistentClient(path=inputs.input_dir)
    collection = client.get_collection(name="default_collection")
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(api_key=os.getenv('OPENAI_API_KEY'))

    query_embedding = openai_ef.embed_with_retries([inputs.question])[0]
    sel_docs = collection.query(query_embedding, n_results=3)
    retrieved_docs_str = get_retrieved_docs_str(sel_docs)

    prompt = cfg["inputs"]["user_message_template"].format(question=inputs.question, document=retrieved_docs_str)

    messages = [
        {
            "role": "system",
            "content": cfg["inputs"]["system_message"]
        },
        {
            "role": "user",
            "content": prompt
        }
    ]

    response = completion(
        model=cfg["models"]["openai"]["model"],
        messages=messages
    )

    return response.choices[0].message.content



if __name__ == "__main__":
    import yaml
    inputs = InputSchema(
        question = "What is the main concept discussed in the paper?",
        input_dir = "/Users/arshath/play/playground/node-tests/chroma_db",
    )
    
    cfg_path = "simple_rag/component.yaml"
    with open(cfg_path, "r") as file:
        cfg = yaml.load(file, Loader=yaml.FullLoader)
    res = run(inputs, cfg=cfg)
    print(res)