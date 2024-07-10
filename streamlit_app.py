import streamlit as st
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.vectorstores.azuresearch import AzureSearch
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from azure.search.documents.indexes import SearchIndexClient
from azure.core.credentials import AzureKeyCredential
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import FlashrankRerank

# Initialize session state
st.session_state.setdefault("messages", [{"role": "assistant", "content": "How can I help you?"}])

# Load configuration from Streamlit secrets
config = st.secrets

# Set up Azure Search
index_client = SearchIndexClient(
    endpoint=config['AZURE_VECTOR_STORE_ENDPOINT'],
    credential=AzureKeyCredential(config['AZURE_VECTOR_STORE_CREDENTIAL']),
)
index = index_client.get_index(config['index_name_query'])
fields = index.fields

# Initialize embeddings and vector stores
azure_embeddings = AzureOpenAIEmbeddings(
    azure_deployment=config['AZURE_OPENAI_API_INSTANCE_NAME_EMB'],
    chunk_size=int(config['embedding_chunk_size']),
)

vector_store_queries = AzureSearch(
    azure_search_endpoint=config['AZURE_VECTOR_STORE_ENDPOINT'],
    azure_search_key=config['AZURE_VECTOR_STORE_CREDENTIAL'],
    index_name=config['index_name_query'],
    embedding_function=azure_embeddings.embed_query,
    fields=fields,
)

def get_query_topic(query, vector_store):
    docs = vector_store.similarity_search(query=query, k=1, search_type="similarity")
    return docs[0].metadata['topic']

def get_db_retriever(topic):
    query_embeddings = HuggingFaceInstructEmbeddings(
        query_instruction=f"Represent the {topic} query for retrieving: "
    )
    vector_store_docs = AzureSearch(
        azure_search_endpoint=config['AZURE_VECTOR_STORE_ENDPOINT'],
        azure_search_key=config['AZURE_VECTOR_STORE_CREDENTIAL'],
        index_name=config['index_name_docs'],
        embedding_function=query_embeddings.embed_query,
        fields=fields,
    )
    return vector_store_docs.as_retriever(k=4, type='hybrid')

# Streamlit UI
st.title('🦜🔗 Instruct embedding App')

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    
    with st.spinner('Searching...'):
        try:
            llm = AzureChatOpenAI(deployment_name=config['AZURE_OPENAI_DEPLOYMENT_NAME'], temperature=0)
            
            system_prompt = """You are an assistant for question-answering tasks. 
            Use the following pieces of most relevant retrieved context to answer 
            the question. If you don't know the answer, say that you 
            don't know. Use three sentences maximum and keep the 
            answer concise.
            
            {context}"""
            
            llm_prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human", "{input}"),
            ])
            
            question_answer_chain = create_stuff_documents_chain(llm, llm_prompt)
            topic = get_query_topic(prompt, vector_store_queries)
            retriever = get_db_retriever(topic)
            compression_retriever = ContextualCompressionRetriever(
                base_compressor=FlashrankRerank(),
                base_retriever=retriever
            )
            rag_chain = create_retrieval_chain(compression_retriever, question_answer_chain)
            results = rag_chain.invoke({"input": prompt})['answer']
            
            st.session_state.messages.append({"role": "assistant", "content": results})
            st.chat_message("assistant").write(results)
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
