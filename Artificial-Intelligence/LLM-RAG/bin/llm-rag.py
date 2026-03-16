#!/usr/bin/env python3

import textwrap
import os
import sys

#from langchain_community.chat_models import ChatOllama
from langchain_ollama import ChatOllama
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain.schema.output_parser import StrOutputParser
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.runnable import RunnablePassthrough
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores.utils import filter_complex_metadata



def wrap(body):
    # https://stackoverflow.com/a/26538082
    body = '\n'.join(['\n'.join(textwrap.wrap(line, 70,
                 break_long_words=False, replace_whitespace=False))
                 for line in body.splitlines()])
    return body


def ask(prompt, model=None, chain=None):
    print("\n")
    print(f"{prompt}\n")
    if model:
        answer = model.invoke(prompt).content
    else:
        answer = chain.invoke(prompt)
    print(wrap(answer))
    print("\n")
    
    
book_fname = sys.argv[1]
    
print("\n===============================================")
print(f"Book: {book_fname}")


# llm
model = ChatOllama(model="mistral", \
                   base_url="http://127.0.0.1:"+os.getenv("OLLAMA_PORT", "11434"))

prompt_rag =  """
              <s> [INST] You are an expert book summarizer. You are answering questions about the content in the book. Only use the following context to answer the question.[/INST] </s>
              [INST] Question: {question}
              Context: {context}
              Answer: [/INST]
              """

# first test the model
ask("Please tell me what kind of LLM you are, and describe what data you were trained on.", model=model)

# for the RAG example, we have to read the book, convert it to chunks
prompt = PromptTemplate.from_template(prompt_rag)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=100)

docs = []
docs.extend(TextLoader(file_path=book_fname).load())

chunks = text_splitter.split_documents(docs)
chunks = filter_complex_metadata(chunks)

vector_store = Chroma.from_documents(documents=chunks, embedding=FastEmbedEmbeddings())
retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            "k": 3,
            "score_threshold": 0.2,
            },
        )

chain = ({"context": retriever, "question": RunnablePassthrough()}
                | prompt
                | model
                | StrOutputParser())

# now ask the question using the augmented model (RAG)
# note that we are passing the rag chain, instead of the just the model

ask("Please provide a one paragraph summary of the book.", chain=chain)

ask("Who is the protagonist in the book?", chain=chain)

ask("Who is the antagonist in the book?", chain=chain)

ask("What time period is the book set in?", chain=chain)


