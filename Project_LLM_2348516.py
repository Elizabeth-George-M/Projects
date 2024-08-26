#!/usr/bin/env python
# coding: utf-8

# # Project Description 

# This project offers a comprehensive solution for processing articles from web URLs and providing insightful responses through a chatbot. By utilizing a pipeline that extracts and analyzes content from the articles, the system enables users to interact with the information in a dynamic way.
# 
# The chatbot component allows users to ask questions like "Can we invest?" or "Are the stock prices high?" based on the content of the article. This conversational interface enhances user engagement by offering quick, relevant answers derived from the processed text. The system’s design makes it especially valuable for applications in financial research, investment analysis, or any field where users need to extract actionable insights from articles in real-time.
# 
# With the ability to handle various queries about the article, the chatbot serves as a useful tool for gaining deeper understanding and making informed decisions based on the provided content.

# # Packages to be installed

# In[25]:


get_ipython().system('pip install langchain-community')


# In[26]:


get_ipython().system('pip install langchain-community')


# In[27]:


get_ipython().system('pip install langchain-community')


# In[28]:


import nltk
nltk.download('punkt')


# In[29]:


get_ipython().system('pip install faiss-cpu')


# In[30]:


get_ipython().system('pip install langchain')


# In[31]:


get_ipython().system('pip install langchain transformers')


# In[ ]:





# # Libraries
# 

# In[14]:


import os
import pickle
import time
import langchain
from langchain import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_core.embeddings import FakeEmbeddings
from transformers import pipeline
from langchain.llms import HuggingFacePipeline
from concurrent.futures import ThreadPoolExecutor
import re


# # Preprocessing

# In[15]:


# Preprocessing function for text cleaning
def preprocess_text(text, lowercase=True, remove_special_chars=True):
    if lowercase:
        text = text.lower()
    if remove_special_chars:
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text

# Hugging Face model pipeline
pipe = pipeline("text2text-generation", model="google/flan-t5-small")
hf = HuggingFacePipeline.from_model_id(
    model_id="google/flan-t5-small",
    task="text2text-generation",
    pipeline_kwargs={"max_new_tokens": 500},
)


# # URL's

# In[16]:


# URLs to load
urls = [
    "https://www.moneycontrol.com/technology/a-win-win-for-paytm-zomato-shareholders-the-deal-holds-promise-for-both-ncr-firms-article-12803827.html",
    "https://www.fool.com/investing/2024/08/24/nvidia-stock-soared-30-since-stock-split-this-next/"
]


# # Loading the data

# In[17]:


# Load documents concurrently
def load_documents_concurrently(url_list):
    with ThreadPoolExecutor() as executor:
        loaders = [UnstructuredURLLoader(urls=[url]) for url in url_list]
        data = list(executor.map(lambda loader: loader.load(), loaders))
        # Flatten the list of lists
        data = [item for sublist in data for item in sublist]
        return data

data = load_documents_concurrently(urls)


# Ensure documents are loaded
print(f"Number of loaded documents: {len(data)}")


# # Model

# In[18]:


# Process documents with Hugging Face model and clean text
for i, document in enumerate(data):
    if document:
        # Preprocess text before passing to the model
        processed_content = preprocess_text(document.page_content)
        print(f"\nDocument {i+1} (Processed):\n{processed_content[:500]}...")

        # Generate a response using the Hugging Face model
        response = hf(processed_content)
        print(f"Generated Text: {response}")


# # Processing the data

# In[19]:


# Split documents into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=20
)

# Check and split loaded documents
docs = []
for i, doc in enumerate(data):
    if doc and doc.page_content.strip():  # Only split non-empty documents
        chunks = text_splitter.split_documents([doc])
        docs.extend(chunks)
    else:
        print(f"Skipping empty or invalid document at index {i}")

print(f"Total number of chunks: {len(docs)}")


# # Processing the text from the document 

# In[20]:


# Fake embeddings (for demo purposes)
embeddings = FakeEmbeddings(size=100)

# Create FAISS index and add custom metadata
vector_index_hugging_face = FAISS.from_documents(docs, embeddings)


# # Storing the processed text

# In[21]:


# Persist the FAISS index using pickle
file_path = "vector_index.pkl"
with open(file_path, "wb") as f:
    pickle.dump(vector_index_hugging_face, f)

absolute_file_path = os.path.abspath(file_path)
print(f"File saved at: {absolute_file_path}")

# Load the persisted FAISS index from file if exists
if os.path.exists(file_path):
    with open(file_path, "rb") as f:
        vectorindex = pickle.load(f)
    print("Loaded FAISS index from file.")


# # Defining the parameters for the model for Q and A

# In[22]:


chain = RetrievalQAWithSourcesChain.from_llm(llm=hf, retriever=vectorindex.as_retriever())


# # Query Handling

# In[23]:


# Query handling
query = 'What is this article about?'

langchain.debug = True
start_time = time.time()
results = chain({"question": query}, return_only_outputs=True)
end_time = time.time()


# # Displaying the output

# In[24]:


# Display results with processing time
print(f"Answer: {results['output_text']}")
print(f"Sources: {results['source_documents']}")
print(f"Processed in {end_time - start_time:.2f} seconds")


# In[ ]:






















# In[ ]:




