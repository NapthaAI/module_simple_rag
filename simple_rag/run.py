from simple_rag.schemas import InputSchema
from naptha_sdk.utils import get_logger
from typing import Dict
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate


logger = get_logger(__name__)


def get_retrieved_docs_str(retrieved_docs):
    return '\n'.join([f'Doc {i}:\n{doc.page_content}\n\n' for i, doc in enumerate(retrieved_docs)])

def run(inputs: InputSchema, worker_nodes = None, orchestrator_node = None, flow_run = None, cfg: Dict = None):
    logger.info(f"Running module with prompt: {inputs.question}")

    chroma_db = Chroma(
        embedding_function=OpenAIEmbeddings(),
        persist_directory=inputs.input_dir,
    )

    retrieved_docs = chroma_db.similarity_search(inputs.question, k=4)
    retrieved_docs_str = get_retrieved_docs_str(retrieved_docs)

    messages = [
        ("system", cfg["inputs"]["system_message"]),
        ("user", cfg["inputs"]["user_message_template"]),
    ]

    chat_template = ChatPromptTemplate.from_messages(messages, template_format="mustache")
    chat_prompt = chat_template.invoke({
        "question": inputs.question,
        "document": retrieved_docs_str,
    })

    llm = ChatOpenAI(model='gpt-3.5-turbo')

    response = llm.invoke(chat_prompt.to_messages())

    return response.content


if __name__ == "__main__":
    inputs = InputSchema.parse_obj({
        "question": "What is the main concept discussed in the paper?",
        "input_dir": "/Users/arshath/play/playground/node-tests/chroma_db",
    })
    
    res = run(inputs)
    print(res)