from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA
import os
import time
start = time.time()

DATA_FOLDER = './folder_name'


def pdf_loader(data_folder=DATA_FOLDER):
    pdf_files = [fn for fn in os.listdir(data_folder) if fn.endswith('.pdf')]
    loaders = [PyPDFLoader(os.path.join(data_folder, fn)) for fn in pdf_files]
    print(f'{len(loaders)} files loaded')
    return loaders

# Load multiple PDF documents using the pdf_loader function
loaders = pdf_loader()

# Combine the loaded documents from different loaders
documents = []
for loader in loaders:
    documents.extend(loader.load())

tokenizer = AutoTokenizer.from_pretrained("lmsys/fastchat-t5-3b-v1.0")
model = AutoModelForSeq2SeqLM.from_pretrained("lmsys/fastchat-t5-3b-v1.0")
pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer, max_length=256)
local_llm = HuggingFacePipeline(pipeline=pipe)
embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")

# Get your splitter ready
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

# Split your docs into texts
texts = text_splitter.split_documents(documents)
print("embedding....")
# Embed your texts
db = FAISS.from_documents(texts, embeddings)
db.save_local("saved-embeddings")

emb_time = time.time()
print("embedding took:", emb_time - start)