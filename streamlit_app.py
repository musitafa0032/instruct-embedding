import streamlit as st
# import langchain.prompts.prompt
# import langchain_community.chat_models
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.vectorstores.azuresearch import AzureSearch
from langchain_openai import AzureOpenAIEmbeddings
from azure.search.documents.indexes import SearchIndexClient
from azure.core.credentials import AzureKeyCredential
from langchain_openai import AzureChatOpenAI
# from langchain.chains import RetrievalQA
# from langchain.prompts import PromptTemplate
import os
import textwrap
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import FlashrankRerank
from dotenv import load_dotenv
# from langchain_community.document_compressors.rankllm_rerank import RankLLMRerank



st.title('🦜🔗 Instruct embedding App')

if "messages" not in st.session_state:
      st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

for msg in st.session_state.messages:
      st.chat_message(msg["role"]).write(msg["content"])

AZURE_VECTOR_STORE_ENDPOINT=st.secrets['AZURE_VECTOR_STORE_ENDPOINT']
AZURE_VECTOR_STORE_CREDENTIAL = st.secrets['AZURE_VECTOR_STORE_CREDENTIAL']
index_name_query = st.secrets['index_name_query']
index_name_docs = st.secrets['index_name_docs']
AZURE_OPENAI_API_INSTANCE_NAME_EMB = st.secrets['AZURE_OPENAI_API_INSTANCE_NAME_EMB']
embedding_chunk_size=int(st.secrets['embedding_chunk_size'])
os.environ["AZURE_OPENAI_API_KEY"] = st.secrets['AZURE_OPENAI_API_KEY']
os.environ["AZURE_OPENAI_ENDPOINT"]= st.secrets['AZURE_OPENAI_ENDPOINT']
AZURE_OPENAI_DEPLOYMENT_NAME = st.secrets['AZURE_OPENAI_DEPLOYMENT_NAME']
os.environ['OPENAI_API_VERSION'] = st.secrets['OPENAI_API_VERSION']

index_client = SearchIndexClient(
        endpoint=AZURE_VECTOR_STORE_ENDPOINT,
        credential=AzureKeyCredential(AZURE_VECTOR_STORE_CREDENTIAL),
    )
index = index_client.get_index(index_name_query)

azure_embeddings = AzureOpenAIEmbeddings(
        azure_deployment=AZURE_OPENAI_API_INSTANCE_NAME_EMB,
        chunk_size=embedding_chunk_size,
    )

def get_query_topic(query,vector_store_queries):
    docs = vector_store_queries.similarity_search(
        query=query,
        k=3,
        search_type="similarity",
    )
    topic=docs[0].metadata['topic']
    return topic

def get_db_retriever(topic):
    query_embeddings = HuggingFaceInstructEmbeddings(
                    query_instruction=f"Represent the {topic} query for retrieving: "
                )
    vector_store_docs = AzureSearch(
                    azure_search_endpoint=AZURE_VECTOR_STORE_ENDPOINT,
                    azure_search_key=AZURE_VECTOR_STORE_CREDENTIAL,
                    index_name=index_name_docs,
                    embedding_function=query_embeddings.embed_query,
                    fields=fields,
                )
    retriever=vector_store_docs.as_retriever(k=4,type='hybrid')
    return retriever

def wrap_text_preserve_newlines(text, width=110):
    # Split the input text into lines based on newline characters
    lines = text.split('\n')

    # Wrap each line individually
    wrapped_lines = [textwrap.fill(line, width=width) for line in lines]

    # Join the wrapped lines back together using newline characters
    wrapped_text = '\n'.join(wrapped_lines)

    return wrapped_text
def process_llm_response(llm_response):
    print(wrap_text_preserve_newlines(llm_response['result']))
    print('\n\nSources:')
    for source in llm_response["source_documents"]:
        print(source.metadata['source'])
        

system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of most relevant retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

fields=index.fields
vector_store_queries = AzureSearch(
                azure_search_endpoint=AZURE_VECTOR_STORE_ENDPOINT,
                azure_search_key=AZURE_VECTOR_STORE_CREDENTIAL,
                index_name=index_name_query,
                embedding_function=azure_embeddings.embed_query,
                fields=fields,
            )

llm=AzureChatOpenAI(deployment_name=AZURE_OPENAI_DEPLOYMENT_NAME,temperature=0)
question_answer_chain = create_stuff_documents_chain(llm, prompt)

query = st.text_input("Please input your question")
topic=get_query_topic(query,vector_store_queries)
retriever=get_db_retriever(topic)

compressor = FlashrankRerank()
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, base_retriever=retriever
)
rag_chain = create_retrieval_chain(compression_retriever, question_answer_chain)
results=rag_chain.invoke({"input": query})['answer']
      
st.session_state.messages.append({"role": "assistant", "content": results})
st.chat_message("assistant").write(results)

