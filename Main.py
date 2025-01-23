from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader

from langchain.text_splitter import RecursiveCharacterTextSplitter

data_path='data/'
db_faiss_path='vectorstore/db_faiss'

#create vector databse
def create_vector_db():
    loader=DirectoryLoader(data_path, glob='*.pdf',loader_cls=PyPDFLoader)
    documents=loader.load()
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=50)
    texts=text_splitter.split_documents(documents)
    embeddings=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',model_kwargs={'device':'cpu'})
    db=FAISS.from_documents(texts,embeddings)
    db.save_local(db_faiss_path)

if  __name__=="__main__":
    create_vector_db()
