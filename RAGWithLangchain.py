from langchain.chains import RetrievalQA
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_community.llms import Ollama
from langchain_community.vectorstores import FAISS
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_text_splitters import CharacterTextSplitter

from langchain_community.document_loaders import PyPDFLoader
loader = PyPDFLoader(r"C:\Users\Dell\OneDrive\Desktop\PythonApplications\RAG\data\RAMAKRISHNA_CV.pdf")


#loader = UnstructuredFileLoader(r"C:\Users\Dell\OneDrive\Desktop\PythonApplications\RAG\data\RAMAKRISHNA_CV.pdf")
docs = loader.load()

text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=2000,
    chunk_overlap=200
)
texts = text_splitter.split_documents(docs)

# embeddings = HuggingFaceEmbeddings()
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


db = FAISS.from_documents(texts, embeddings)

llm = Ollama(model="llama3")
# llm = Ollama(model="llama3", base_url="<a href="http://localhost:11434">http://localhost:11434</a>")


chain = RetrievalQA.from_chain_type(
    llm,
    retriever=db.as_retriever()
)

# question = "Can you please summarize the document"
question = "Can you provide the eduction details of the person?"
result = chain.invoke({"query": question})

print(result['result'])