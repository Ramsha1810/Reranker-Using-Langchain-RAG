from langchain_openai import ChatOpenAI
from qdrant_client import QdrantClient
from langchain_qdrant import Qdrant
from langchain_community.vectorstores import Qdrant
from datetime import datetime
import os
from pathlib import Path
import tempfile
from dotenv import load_dotenv
import logging
import os
from langchain.chains import RetrievalQA
from langchain_qdrant import Qdrant
from langchain_community.document_loaders import  Docx2txtLoader,PyMuPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.docstore.document import Document
from pathlib import Path
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import  RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain.schema import format_document
# from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_core.messages import AIMessage, HumanMessage, get_buffer_string
from operator import itemgetter
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import FlashrankRerank
from langchain.prompts import PromptTemplate




logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter(
    '%(asctime)s:%(name)s:%(levelname)s:%(message)s:%(funcName)s')
file_handler = logging.FileHandler('s3_draft_missing.log')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)
load_dotenv()

class IncomingFileProcessor():
    def __init__(self, chunk_size=750) -> None:
        self.chunk_size = chunk_size
    
    def get_pdf_splits(self, pdf_file: str, filename: str):
        try:
            loader = PyMuPDFLoader(pdf_file)
            pages = loader.load()
            logger.info("Succesfully loaded the pdf file")
            textsplit = RecursiveCharacterTextSplitter(
                separators=["\n\n",".","\n"],
                chunk_size=self.chunk_size, chunk_overlap=15, 
                length_function=len)
            doc_list = []
            for pg in pages:
                pg_splits = textsplit.split_text(pg.page_content)
                for page_sub_split in pg_splits:
                    metadata = {"source": filename}
                    doc_string = Document(page_content=page_sub_split, metadata=metadata)
                    doc_list.append(doc_string)
            logger.info("Succesfully split the pdf file")
            return doc_list
            # return pages
        except Exception as e:
            logger.critical(f"Error in Loading pdf file: {str(e)}")
            raise Exception(str(e))
           
    def get_docx_splits(self, docx_file: str, filename: str):
        try:
            loader = Docx2txtLoader(str(docx_file))
            txt = loader.load()
            logger.info("Succesfully loaded the docx file")
            textsplit = RecursiveCharacterTextSplitter(
                separators=["\n\n",".","\n"],
                chunk_size=self.chunk_size, chunk_overlap=15, 
                length_function=len)

            doc_list = textsplit.split_text(txt[0].page_content)
            new_doc_list = []
            for page_sub_split in doc_list:
                metadata = {"source": filename}
                doc_string = Document(page_content=page_sub_split, metadata=metadata)
                new_doc_list.append(doc_string)
            logger.info("Succesfully split the docx file")
            return new_doc_list
        except Exception as e:
            logger.critical("Error in Loading docx file:"+str(e))
            raise Exception(str(e))
        
    def get_txt_split(self, txt_file: str, filename:str):
        try: 
            loader  =  TextLoader(txt_file)
            print(loader)
            txt = loader.load()
            logger.info("Successfully loaded the text file")
            textsplit = RecursiveCharacterTextSplitter(
                separators=["\n\n",".","\n"],
                chunk_size=self.chunk_size, chunk_overlap=15, 
                length_function=len)
            text_list = textsplit.split_text(txt[0].page_content)
            new_text_list  = []
            for page_sub_split in text_list:
                metadata = {"source": filename}
                text_string = Document( page_content=page_sub_split, metadata=metadata)
                new_text_list.append(text_string)
            logger.info("Successfully split the text file")
            return new_text_list
        except Exception as e:
            logger.critical("Error loading text file:"+str(e))
            raise Exception(str(e))


def load_local_vectordb_using_qdrant(vectordb_folder_path, embed_fn):
    qdrant_client = QdrantClient(
        url=os.getenv('qdrant_url'), 
        prefer_grpc=True,
        api_key=os.getenv('qdrant_api_key'),

    )
    qdrant_store= Qdrant(qdrant_client, vectordb_folder_path, embed_fn)
    return qdrant_store  

# ------------------------------
def semantic_search_conversation(query, vectorstore):
    try:
        num_chunks= 10
        
        model = ChatOpenAI(model = "gpt-3.5-turbo-16k", openai_api_key = os.getenv("OPENAI_API_KEY"), temperature=0.3)
        retriever_=vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": num_chunks})
        template = """Answer the question based only on the following context:
        {context}

        Question: {question}
        """
        prompt = ChatPromptTemplate.from_template(template)

        compressor = FlashrankRerank()
        setup_and_retrieval = ContextualCompressionRetriever(
            base_compressor=compressor, base_retriever=retriever_
        )
        
        output_parser= StrOutputParser()
        #
        # selective_docs= setup_and_retrieval.invoke(query)
        chain = RetrievalQA.from_chain_type(llm=model, retriever = setup_and_retrieval)
        model_answer= chain.invoke(query)
        print(model_answer)
        response= output_parser.invoke(model_answer['result'])
        return response

    except Exception as e:
        print(f"An error occurred: {e}")
        raise Exception('OpenAI key error {e}')   


def handling_files(contents, file_extension, original_filename):
    file_processor = IncomingFileProcessor(chunk_size=512)
    try:
        if contents:
            if file_extension.endswith('docx'):
                suffix = ".docx"
            elif file_extension.endswith('pdf'):
                suffix = ".pdf"
            elif file_extension.endswith('txt'):
                suffix = ".txt"

            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
                tmp_file.write(contents)

            try:
                if file_extension.endswith('docx'):
                    texts = file_processor.get_docx_splits(tmp_file.name, original_filename)
                elif file_extension.endswith('.pdf'):
                    texts = file_processor.get_pdf_splits(tmp_file.name, original_filename)
                elif file_extension.endswith('.txt'):
                    texts = file_processor.get_txt_split(tmp_file.name, original_filename)
            except Exception as e:
                print(e)
            finally:
                Path(tmp_file.name).unlink()
            return texts
    except Exception as e:
        raise e


def background_task(doc_list, embed_fn,filename):
    try:
        date_ = str(datetime.now()).split(' ')[0]
        cl_name = filename+date_
        qdrant = Qdrant.from_documents(
            documents=doc_list,
            embedding=embed_fn,
            url=os.getenv('qdrant_url'),
            # prefer_grpc=True,
            api_key=os.getenv('qdrant_api_key'),
            collection_name=cl_name,
        )
        print(cl_name)
        logger.info("Successfully created the vectordb")
    except Exception as e: 
        print(e)
        raise Exception(e)
    return cl_name

def conversation_retrieval_chain(query, vectordb):
  num_chunks = 10

  _template = """Answer the question based only on the following context:
  {context}

  Question: {question}
  """
  ANSWER_PROMPT = ChatPromptTemplate.from_template(_template)
  DEFAULT_DOCUMENT_PROMPT = PromptTemplate.from_template(template="{page_content}")
  model = ChatOpenAI(model = "gpt-3.5-turbo-16k", openai_api_key = os.getenv("OPENAI_API_KEY"), temperature=0.3)
  retriever_=vectordb.as_retriever(search_type="similarity", search_kwargs={"k": num_chunks})

  compressor = FlashrankRerank()
  setup_and_retrieval = ContextualCompressionRetriever(
    base_compressor=compressor, base_retriever=retriever_
    )
  
#   retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": num_chunks})
  def _combine_documents(
      docs, document_prompt=DEFAULT_DOCUMENT_PROMPT, document_separator="\n\n"
  ):
    doc_strings = [format_document(doc, document_prompt) for doc in docs]
    return document_separator.join(doc_strings)
  _context = {
        "context": _combine_documents(retriever_.invoke(query)),
        "question": query,
    }
  
  chain = RetrievalQA.from_chain_type(llm=model, retriever = setup_and_retrieval)
  model_answer= chain.invoke(query)
  output_parser = StrOutputParser()
  response = output_parser.invoke(model_answer['result'])
  return response

chat_history = [
    HumanMessage(content="What is the type of agreement?"),
    AIMessage(content="Commercial lease"),
]






# def conversation_retrieval_chain(query, vectordb):
#     num_chunks= 10
#     _template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.

#     Chat History:
#     {chat_history}
#     Follow Up Input: {question}
#     Standalone question:"""
#     CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)
#     template = """Answer the question based only on the following context:
#     {context}

#     Question: {question}
#     """
#     ANSWER_PROMPT = ChatPromptTemplate.from_template(template)
#     DEFAULT_DOCUMENT_PROMPT = PromptTemplate.from_template(template="{page_content}")
#     retriever= vectordb.as_retriever(search_type="similarity", search_kwargs={"k": num_chunks})
    
#     def _combine_documents(
#         docs, document_prompt=DEFAULT_DOCUMENT_PROMPT, document_separator="\n\n"
#     ):
#         doc_strings = [format_document(doc, document_prompt) for doc in docs]
#         return document_separator.join(doc_strings)
#     _inputs = RunnableParallel(
#     standalone_question=RunnablePassthrough.assign(
#         chat_history=lambda x: get_buffer_string(x["chat_history"])
#     )
#     | CONDENSE_QUESTION_PROMPT
#     | ChatOpenAI(temperature=0)
#     | StrOutputParser(),
#     )
#     _context = {
#         "context": itemgetter("standalone_question") | retriever | _combine_documents,
#         "question": lambda x: x["standalone_question"],
#     }
#     # conversational_qa_chain = _inputs | _context | ANSWER_PROMPT | ChatOpenAI()
#     rephrase_question= _inputs.invoke(        {
#             "question": query,
#             "chat_history": [
#                 HumanMessage(content="What is the type of agreement?"),
#                 AIMessage(content="Commercial lease"),
#             ],
#         })
#     setup_and_retrieval = RunnableParallel(
#         {"context": retriever, "question": RunnablePassthrough()}
#     )
#     context= setup_and_retrieval.invoke(rephrase_question['standalone_question'])
#     docs= _combine_documents(context['context'])
#     doc_dict= {'context':docs, "question": rephrase_question['standalone_question']}
#     answer_prompt= ANSWER_PROMPT.invoke(doc_dict)
#     output_parser= StrOutputParser()
#     final_response= ChatOpenAI().invoke(answer_prompt)
#     response= output_parser.invoke(final_response)

#     return response


