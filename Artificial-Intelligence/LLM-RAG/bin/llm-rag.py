#!/usr/bin/env python3

import textwrap
import os

from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain.schema.output_parser import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.runnable import RunnablePassthrough
from langchain.prompts import PromptTemplate
from langchain.vectorstores.utils import filter_complex_metadata



def wrap(body):
    # https://stackoverflow.com/a/26538082
    body = '\n'.join(['\n'.join(textwrap.wrap(line, 70,
                 break_long_words=False, replace_whitespace=False))
                 for line in body.splitlines()])
    return body



# llm
model = ChatOllama(model="mistral", \
                   base_url="http://127.0.0.1:"+os.getenv("OLLAMA_PORT", "11434"))

# Be succinct.
instructions_regular = "You are an assistant for question-answering tasks. If you don't know the answer, just say that you don't know."
prompt_rag =  """
              <s> [INST] You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. [/INST] </s>
              [INST] Question: {question}
              Context: {context}
              Answer: [/INST]
              """

# first test the model
print("\n\n\n===============================================")
print("LLM details, as described by itself:\n")
answer = model.invoke("Please tell me what kind of LLM you are, and describe what data you were trained on.").content
print(wrap(answer))

# ask the question using only the knowledge built in the trained model
print("\n\n\n===============================================")
print("Summary without RAG:\n")
answer = model.invoke(f"{instructions_regular} Please provide a summary of the GPU capabilities of the IU Jetstream2 system. Include the instance flavors, and any details about the GPUs.").content
print(wrap(answer))

# for the RAG example, we have to read the PDF, convert it to chunks
prompt = PromptTemplate.from_template(prompt_rag)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=100)

docs = []
for file in os.listdir("."):
    if file.endswith('.pdf'):
        docs.extend(PyPDFLoader(file_path=file).load())

chunks = text_splitter.split_documents(docs)
chunks = filter_complex_metadata(chunks)

vector_store = Chroma.from_documents(documents=chunks, embedding=FastEmbedEmbeddings())
retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            "k": 3,
            "score_threshold": 0.5,
            },
        )

chain = ({"context": retriever, "question": RunnablePassthrough()}
                | prompt
                | model
                | StrOutputParser())

# now ask the question using the augmented model (RAG)
print("\n\n\n===============================================")
print("Summary with RAG:\n")
answer = chain.invoke("Please provide a summary of the GPU capabilities of the IU Jetstream2 system. Include the instance flavors, and any details about the GPUs.")
print(wrap(answer))

print("\n\n\n===============================================")

