from typing import List
import os
from operator import itemgetter
from dotenv import load_dotenv

from langchain_postgres import PGVector
from langchain_postgres.vectorstores import PGVector
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage
from langchain_core.runnables import ConfigurableFieldSpec, RunnableBranch
from langchain.retrievers import ParentDocumentRetriever
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain.storage._lc_store import create_kv_docstore
from langchain.storage import LocalFileStore
from langchain.globals import set_debug
from langchain.retrievers.multi_vector import SearchType
from langchain_community.document_loaders import DirectoryLoader, TextLoader

set_debug(False)


connection_string = "postgresql+psycopg://tramyho:1231@localhost:5432/postgres"
collection_name = "postgres"
load_dotenv()


def setup():
    # CREATE EMBEDDINGS AND VECTOR DB
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    chat_model = ChatOpenAI(
        model="gpt-4.1",
        temperature=0,    
    )

    TEMPLATE = """\
    You are a helpful, friendly educational assistant designed to support primary school students. 

    Your job is to answer their questions **only using information from a provided document** (a knowledge base about cells, chemistry of life, and related biology topics). You must only answer questions if they are within the scope of that knowledge base. 

    If a question cannot be answered based on the document, respond politely with:  
    **"I'm not sure how to answer that based on the information I have."**

    🧠 Requirements:
    - Use information retrieved from the document to generate **accurate and grounded answers**.
    - Always keep your answers **simple, clear, and age-appropriate** for primary school learners.
    - Use easy-to-understand explanations, short sentences, and avoid scientific jargon unless explained in simple terms.
    - You may use analogies, fun facts, or emojis to help kids better understand.

    🌍 Language Support:
    - If the question is asked in another language (e.g., Mandarin, Malay), respond in that same language after translating and understanding it. Keep the tone friendly and suitable for kids.

    Examples:
    - ❌ Don’t make up facts.
    - ✅ Do explain concepts like “cell membrane” as “the skin of a cell that controls what goes in and out.”

    Stay cheerful, educational, and always on-topic.


    Query: {question}

    Context: {context}
    """

    return embeddings, chat_model, TEMPLATE


embeddings, chat_model, TEMPLATE = setup()
# folder_path = '9000093355'
# file_paths = ['9000093355/' + f for f in os.listdir(folder_path) if f.lower().endswith('.pdf')]

loader = DirectoryLoader("data", glob="*.md", show_progress=True, loader_cls=TextLoader )
documents = loader.load()
# PARENT RETRIEVER
vectorstore = PGVector(
    embeddings=embeddings,
    collection_name=collection_name,
    connection=connection_string,
    pre_delete_collection=True,
)

# splitter = RecursiveCharacterTextSplitter(chunk_size=2048, chunk_overlap=256)
# docs = splitter.split_documents(documents)


class InMemoryHistory(BaseChatMessageHistory, BaseModel):
    messages: List[BaseMessage] = Field(default_factory=list)

    def add_messages(self, messages: List[BaseMessage]) -> None:
        self.messages.extend(messages)

    def clear(self) -> None:
        self.messages = []


def format_docs(docs):
    # for i in docs:
    #     print(i)
    #     print("\n")
    return "\n\n".join(doc.page_content for doc in docs)


def get_by_session_id(session_id: str, store: dict) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryHistory()
    return store[session_id]


def create_parent_chain(vectorstore, store, parent_store):

    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1024, add_start_index=True, chunk_overlap=128
    )
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2048, add_start_index=True, chunk_overlap=256
    )

    retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=parent_store,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
        search_kwargs={"k": 20},
    )
    retriever.add_documents(documents, ids=None)

    retriever.search_type = SearchType.similarity

    # Create the BM25 keyword retriever
    bm25_retriever = BM25Retriever.from_documents(documents)
    bm25_retriever.k = 20  # Number of documents to retrieve

    retriever = EnsembleRetriever(
        retrievers=[retriever, bm25_retriever],
        weights=[0.6, 0.4]  # Adjust weights based on your needs
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                TEMPLATE,
            ),
            MessagesPlaceholder(variable_name="history"),
            ("user", "{question}"),
        ]
    )

    retriever_prompt = ChatPromptTemplate.from_messages(
        [
            MessagesPlaceholder(variable_name="history"),
            ("user", "{question}"),
            (
                "user",
                "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation.",
            ),
        ]
    )

    retrieverOutputLike = RunnableBranch(
        (
            # Both empty string and empty list evaluate to False
            lambda x: not x.get("history", False),
            # If no chat history, then we just pass input to retriever
            (lambda x: x["question"]) | retriever,
        ),
        # If chat history, then we pass inputs to LLM chain, then to retriever
        retriever_prompt | chat_model | StrOutputParser() | retriever,
    ).with_config(run_name="chat_retriever_chain")

    retrieval_chain = (
        {
            "context": retrieverOutputLike | format_docs,
            "question": itemgetter("question"),
            "history": itemgetter("history"),
        }
        | prompt
        | chat_model
        | StrOutputParser()
    )

    chain_with_history = RunnableWithMessageHistory(
        retrieval_chain,
        lambda session_id: get_by_session_id(session_id, store=store),
        input_messages_key="question",
        history_messages_key="history",
        history_factory_config=[
            ConfigurableFieldSpec(
                id="session_id",
                annotation=str,
                default="",
                is_shared=True,
            )
        ],
    )

    return chain_with_history


def process_chat(chain, question, session_id):
    response = chain.invoke(
        {
            "question": question,
        },
        config={"configurable": {"session_id": session_id}},
    )
    return response

# Pre-retrieval Query Rewriting Function
def query_rewrite(query: str, llm: ChatOpenAI):
    # Rewritten Query Prompt
    query_rewrite_prompt = f"""
      You are a helpful assistant that takes a user's query and
      turns it into a short statement or paragraph so that it can
      be used in a semantic similarity search on a vector database
      to return the most similar chunks of content based on the
      rewritten query. Please make no comments, just return the
      rewritten query.
      Remember to use the same language as the original query.
      The original query is: {query}

      ai: """

    # Invoke LLM
    retrieval_query = llm.invoke(query_rewrite_prompt).content
    return retrieval_query

if __name__ == "__main__":
    store = {}

    fs = LocalFileStore("./store_location")
    parent_store = create_kv_docstore(fs)

    chain = create_parent_chain(vectorstore, store, parent_store)
    session_id = 1

    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break
        session_id = 1
        # user_input = query_rewrite(user_input, chat_model)
        response = process_chat(chain, user_input, session_id)

        print("Assistant:", response)
