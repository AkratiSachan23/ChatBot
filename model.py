from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings

import chainlit as cl

db_faiss_path='vectorstore/db_faiss'


custom_prompt_template="""use given information to answer user's question. If you dont know the answer

Context:{context}
Question:{question}

Only return the helpful answer below and nothing els.
Helpful answer:

"""
def set_custom_prompt():
    """
    Prompt template for QA retrieval for each vectorstore
    """
    prompt=PromptTemplate(template=custom_prompt_template, input_variables=['context','question'])
    return prompt

#retrieval QA Chain
def retrieval_qa_chain(llm,prompt,db):
    qa_chain=RetrievalQA.from_chain_type(llm=llm,
                                         chain_type='stuff',
                                         retriever=db.as_retriever(search_kwargs={'k':2}),
                                         return_source_documents=True,
                                         chain_type_kwargs={'prompt':prompt}
                                         )
    return qa_chain

#loading the model
def load_llm():
    #load the locally downloaded model here
    llm=CTransformers(
        model="TheBloke/Llama-2-7B-Chat-GGML",
        model_type="llama",
        max_new_tokens=512,
        temperature=0.5
    )
    return llm

#QA Model Function
def qa_bot():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                       model_kwargs={'device': 'cpu'})
    db = FAISS.load_local(db_faiss_path, embeddings, allow_dangerous_deserialization=True)
    llm = load_llm()
    qa_prompt = set_custom_prompt()
    qa = retrieval_qa_chain(llm, qa_prompt, db)

    return qa


#output Function
def final_result(query):
    qa_result=qa_bot()
    response=qa_result({'query':query})
    return response

#chainlit Code
@cl.on_chat_start
async def start():
    chain=qa_bot()
    msg=cl.Message(content="Starting the bot...")
    await msg.send()
    msg.content="Query?"
    await msg.update()

    cl.user_session.set("chain",chain)

@cl.on_message
@cl.on_message
async def main(message):
    # Extract the text content from the message
    query = message.content  # Assuming the content attribute contains the text input
    
    # Retrieve the chain instance
    chain = cl.user_session.get("chain")

    # Set up the callback handler
    cb = cl.AsyncLangchainCallbackHandler(
        stream_final_answer=True, answer_prefix_tokens=["FINAL"]
    )
    cb.answer_reached = True

    # Pass the extracted query instead of the message object
    res = await chain.acall({"query": query}, callbacks=[cb])  # Update here
    answer = res["result"]

    # Send the response back as a message
    cl.Message(content=answer).send()


