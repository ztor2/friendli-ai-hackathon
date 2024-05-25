import gradio as gr
import os
import random
import requests

from langchain.chains import ConversationalRetrievalChain
from langchain_openai import OpenAIEmbeddings
from langchain_community.chat_models.friendli import ChatFriendli

import qdrant_client
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core import SimpleDirectoryReader
from llama_index.core.indices.multi_modal.base import MultiModalVectorStoreIndex
from llama_index.multi_modal_llms.openai import OpenAIMultiModal


os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_KEY"
os.environ["FRIENDLI_TOKEN"] = "YOUR_FRD_KEY"

llm = ChatFriendli(model="meta-llama-3-70b-instruct")

template = """
Use the following pieces of context to answer the question at the end. If you don’t know the answer, just say that you don’t know, don’t try to make up an answer. Use a combination of context to infer the key points of your answer to make it easier for the user to understand. If a user asks a question in a language other than English, translate your answer to that language as much as possible.
    
{context}
    
Question: {question}
    
Helpful Answer:
"""

client = qdrant_client.QdrantClient(path="qdrant_db") 
image_store = QdrantVectorStore(client=client, collection_name="image_collection")
storage_context = StorageContext.from_defaults(image_store=image_store)
openai_mm_llm = OpenAIMultiModal(model="gpt-4o", max_new_tokens=1500)

def retrieve_contexts(document_ids: list[str], query: str, k: int) -> list[str]:
    resp = requests.post(
        "https://suite.friendli.ai/api/beta/retrieve",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {os.environ["FRIENDLI_TOKEN"]}",
        },
        json={
            "document_ids": document_ids,
            "query": query,
            "k": k,
        }
    )
    data = resp.json()
    return [r["content"] for r in data["results"]]
    
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def get_doc_id(doc_label):
    if doc_label == "LLaVA":
        doc_id = ["SLEVJa39mcbT"]
    elif doc_label == "Interior":
        doc_id = ["zV4Qp4lRXXwg"]
    return doc_id

def response(message, history, doc_label):
    doc_id = get_doc_id(doc_label)

    new_messages = []
    for user, chatbot in history:
        new_messages.append({"role" : "user", "content": user})
        new_messages.append({"role" : "assistant", "content": chatbot})
    new_messages.append({"role": "user", "content": message})

    contexts = retrieve_contexts(doc_id, message, 5)
    rag_message = template.format(context="\n".join(contexts), question=new_messages)
    return llm.call_as_llm(message=rag_message)

def img_retrieve(query, doc_label):
    doc_imgs = SimpleDirectoryReader(f"./{doc_label}").load_data()
    index = MultiModalVectorStoreIndex.from_documents(doc_imgs, storage_context=storage_context)
    img_query_engine = index.as_query_engine(llm=openai_mm_llm,
                                             image_similarity_top_k=3)
    response_mm = img_query_engine.query(query)
    retrieved_imgs = [n.metadata["file_path"] for n in response_mm.metadata["image_nodes"]]
    return retrieved_imgs

with gr.Blocks(theme=gr.themes.Monochrome()) as demo:
    with gr.Row():
        gr.Markdown(
    """
    # 🎨 Multi-modal RAG Chat with FriendliAI
    """)
    with gr.Row():
        gr.Markdown("""Select document from the menu, and interact with the text and images in the document.
                    """)
    with gr.Row():
        with gr.Column(scale=2):       
            doc_label = gr.Dropdown(["LLaVA", "Interior"], label="Select a document:")
            chatbot = gr.ChatInterface(fn=response, additional_inputs=[doc_label], fill_height=True)
        with gr.Column(scale=1):
            sample_1 = "https://friendli.ai/opengraph-image.png"
            sample_2 = "https://friendli.ai/_next/static/media/news-default.6b85ae4e.png"
            sample_3 = "https://s3-us-west-2.amazonaws.com/cbi-image-service-prd/original/6112bbf1-20d1-446a-9950-c75f0c02479e.jpg"
            gallery = gr.Gallery(label="Retrieved images", show_label=True, preview=True, object_fit="contain", value=[(sample_1, 'sample_1'), (sample_2, 'sample_2'), (sample_3, 'sample_3')])
            query = gr.Textbox(label="Enter query")
            button = gr.Button(value="Retrieve images")
            button.click(img_retrieve, [query, doc_label], gallery)

demo.launch(share=True)