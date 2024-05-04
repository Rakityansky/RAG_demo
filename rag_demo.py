import os
import tiktoken
import streamlit as st
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    Settings,
    load_index_from_storage,
)
from llama_index.llms.openai import OpenAI
from llama_index.core.callbacks import CallbackManager, TokenCountingHandler
# from langchain.chat_models import ChatOpenAI

index_name = "./saved_index"
documents_folder = "./documents"
Settings.llm = OpenAI(temperature=0.0, model="gpt-4")
token_counter = TokenCountingHandler(
    tokenizer=tiktoken.encoding_for_model("gpt-4").encode,
    verbose=False,  # set to true to see usage printed to the console
)

Settings.callback_manager = CallbackManager([token_counter])
@st.cache_resource
def initialize_index(index_name, documents_folder):
    if os.path.exists(index_name):
        storage_context = StorageContext.from_defaults(persist_dir=index_name)
        index = load_index_from_storage(storage_context)
    else:
        documents = SimpleDirectoryReader(documents_folder).load_data()
        index = VectorStoreIndex.from_documents(documents)
        index.storage_context.persist(persist_dir=index_name)

    return index


@st.cache_data(max_entries=200, persist=True)
def query_index(_index, query_text):
    if _index is None:
        return "Please initialize the index!"
    response = _index.as_query_engine().query(query_text)
    return str(response)


st.title("PCT Guide Demo")
st.header("Non-production implementation\n single user system\n slow initialization")
st.write(
    "Enter a query about patent application regulations"
)

# index = None
# api_key = st.text_input("Enter your OpenAI API key here:", type="password")
# if api_key:
#     os.environ["OPENAI_API_KEY"] = api_key

index = initialize_index(index_name, documents_folder)


# if index is None:
#     st.warning("Please enter your api key first.")

text = st.text_input("Query text:", value="What is a filing fee in Germany ?")

if st.button("Run Query") and text is not None:
    response = query_index(index, text)
    st.markdown(response)

    llm_col, embed_col = st.columns(2)
    with llm_col:
        st.markdown(
            f"LLM Prompt Tokens Used: {token_counter.prompt_llm_token_count}"
        )
        st.markdown(
            f"LLM Completion Tokens Used: {token_counter.completion_llm_token_count}"
        )

    with embed_col:
        st.markdown(
            f"Embedding Tokens Used: {token_counter.total_embedding_token_count}"
        )
        st.markdown(
            f"Total LLM Tokens Used: {token_counter.total_llm_token_count}"
        )
