import os
import textwrap
import torch
import time
from langchain.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.vectorstores import FAISS  # Change the vector store to Faiss
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import DirectoryLoader
from langchain.chains import RetrievalQA


# Step 1: Load the pre-trained T5 model and tokenizer for LLM
tokenizer = AutoTokenizer.from_pretrained("lmsys/fastchat-t5-3b-v1.0")
model = AutoModelForSeq2SeqLM.from_pretrained("lmsys/fastchat-t5-3b-v1.0")
pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer, max_length=256)
local_llm = HuggingFacePipeline(pipeline=pipe)

# Step 2: Load the instructor embedding model
start = time.time()
instructor_embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
end = time.time()
print("Instructor Embedding model loading time", (end - start))

# Step 3: Load or create the vector database using the Faiss vector store
start = time.time()
embeddings = instructor_embeddings
db = FAISS.load_local("saved-embeddings", embeddings) # load embedding folder
end = time.time()
print("Embeddings from db time", (end - start))

# Step 4: Create the retriever from the vector store
retriever = db.as_retriever()

# Step 5: Create the chain to answer questions using the RetrievalQA
qa_chain = RetrievalQA.from_chain_type(llm=local_llm, chain_type="stuff", retriever=retriever, return_source_documents=True)

# Full example
start = time.time()
query = "Your question"
result = qa_chain({"query": query})
result_value = result['result']
print(result_value)
end = time.time()
print("Query answering time", (end - start))
