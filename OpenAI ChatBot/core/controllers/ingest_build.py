import os
import sys
from langchain import OpenAI, VectorDBQA, LLMChain
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
#from langchain import OpenAI, VectorDBQA
from langchain.document_loaders import DirectoryLoader
from langchain.prompts.prompt import PromptTemplate
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.chains.conversational_retrieval.prompts import CONDENSE_QUESTION_PROMPT
from rich import print
from dotenv import load_dotenv
import openai
load_dotenv()

os.environ['OPENAI_API_KEY'] = "your openai api key"
DATA_PATH = r'C:\Users\ACER\Downloads\OpenAI ChatBot\docs'
MODEL_NAME = 'gpt-3.5-turbo' #gpt-4
CROMADB_DIRECTORY = 'db/'
RETURN_SOURCE_DOCUMENT = True
#llm = ChatOpenAI(temperature=0, model_name=MODEL_NAME)
llm = ChatOpenAI(temperature=0, model_name=MODEL_NAME, max_tokens=1000)

txtLoader = DirectoryLoader(DATA_PATH, glob="**/*.txt", show_progress=True)
pdfLoader = DirectoryLoader(DATA_PATH, glob="**/*.pdf", show_progress=True)
mdLoader = DirectoryLoader(DATA_PATH, glob="**/*.md", show_progress=True)
csvLoader = DirectoryLoader(DATA_PATH, glob="**/*.csv", show_progress=True)
loaders = [txtLoader, pdfLoader, mdLoader, csvLoader]

documents = []
for loader in loaders:
    documents.extend(loader.load())


print(f"Total number of document: {len(documents)}")

textSplitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
documents = textSplitter.split_documents(documents)
print(f"Total number of document after split: {len(documents)}")


print("Creating Vector Store...")
embeddings = OpenAIEmbeddings()
vectorStore = Chroma.from_documents(documents, embeddings , persist_directory=CROMADB_DIRECTORY)
vectorStore.persist()

# metode QA yang bisa digunakan:
# - Similarity Search
# - RetrievalQA
# - RetrievalQA with Prompt
# - ConversationalRetrievalChain

_template = """Diberikan percakapan berikut dan pertanyaan tindak lanjut, ulangi pertanyaan tindak lanjut menjadi pertanyaan mandiri.
Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)

prompt_template = """Gunakan potongan konteks berikut untuk menjawab pertanyaan di bagian akhir. 
Jika Anda tidak tahu jawabannya, katakan saja bahwa Anda tidak tahu, jangan mencoba membuat jawaban.
{context}
Question: {question}
Helpful Answer:"""

"""QA_PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

qa = ConversationalRetrievalChain.from_llm(llm, vectorStore.as_retriever(), CONDENSE_QUESTION_PROMPT, return_source_documents=RETURN_SOURCE_DOCUMENT, combine_docs_chain_kwargs={"prompt": QA_PROMPT})

# Your Question and Answer

chat_history = []


user_message = "sebutkan Undang-Undang Nomor 14 Tahun 1984 tentang Wabah Penyakit Menular beserta sumber !"
#user_message = "jelaskan Undang-Undang Nomor 4 Tahun 1984 ?"
response = qa({"question": user_message, "chat_history": chat_history})
chat_history.append((user_message, response["answer"]))
print(response)
print()
print(response["answer"])


retriever = vectorStore.as_retriever()
qa = RetrievalQA.from_chain_type( llm= OpenAI(), chain_type="stuff", retriever=retriever, return_source_documents=True)
query = "sebutkan Undang-Undang Nomor 14 Tahun 1984 tentang Wabah Penyakit Menular beserta sumber !"
result = qa({"query": query})
print(result)"""

prompt_template_similarity = """""Potongan konteks ini merupakan hasil similiarity search sebelumnya. Jawab pertanyaan dari konteks ini saja. Jika Anda tidak tahu jawabannya katakan saja bahwa Anda tidak tahu, jangan mencoba membuat jawaban. Berikan jawaban dengan lengkap sesuai kalimat/paragraf sumber atau sertakan kata/kalimat pelengkap yang ada dalam kurung.



{context}


Question: {question}
Jawab dalam bahasa indonesia:"""

question= "Gambar Bentuk ankur"
result = vectorStore.similarity_search_with_score(query=question, k=4)
page_content_string = ""
for document_tuple in result:
    document = document_tuple[0]
    page_content = document.page_content
    page_content_string += page_content + "\n" 

llm_chain = LLMChain.from_string(llm=llm, template=prompt_template_similarity)
qa = llm_chain.predict(question=question, context=page_content_string)

source_file = ""

# Retrieve the 'source' field from each metadata dictionary in the list
source_list = [entry[0].metadata['source'] for entry in result]

# Print the 'source' values
for source in source_list:
    source_file += source+ "##" 

response = {
    "result": "",
    "source_file" : ""
}
response["source_file"] = source_file
response["result"] = qa

#print(qa)
  


## delete chroma DB
#vectorStore.delete_collection()
#vectorStore.persist()
