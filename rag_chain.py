import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

# Set OpenRouter config for DeepSeek
os.environ["OPENAI_API_KEY"] = os.getenv("OPENROUTER_API_KEY")
os.environ["OPENAI_API_BASE"] = "https://openrouter.ai/api/v1"
os.environ["OPENAI_API_MODEL"] = "deepseek-chat"

# Load and split PDF
loader = PyPDFLoader("laptop_data.pdf")
pages = loader.load_and_split()

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
documents = text_splitter.split_documents(pages)

# Embedding model
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Vectorstore
vectorstore = Chroma.from_documents(documents, embeddings, persist_directory="chroma_db")
vectorstore.persist()
retriever = vectorstore.as_retriever()

# LLM
llm = ChatOpenAI(model="deepseek/deepseek-r1:free", temperature=0)

# QA Chain
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

def rag_answer(question: str) -> str:
    return qa_chain.run(question)

