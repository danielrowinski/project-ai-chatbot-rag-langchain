import os
from lib import utils
from lib.streaming import StreamHandler
import streamlit as st
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY is not set in .env or environment variables")
print(f"Załadowany klucz API: {api_key}")

from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import SystemMessagePromptTemplate, ChatPromptTemplate

st.set_page_config(page_title="Aplikacja AI Chat", page_icon="\U0001F4D1")
st.title('Ulepszony AI Chat \U0001F916')
st.markdown('Ta aplikacja wykorzystuje AI do odpowiadania na pytania na podstawie załadowanych dokumentów. Wybierz temat i rozpocznij eksplorację!')

class CustomDocChatbot:

    def __init__(self):
        utils.sync_st_session()
        self.llm = utils.configure_llm()
        self.embedding_model = utils.configure_embedding_model()

    @st.spinner('Analizowanie dokumentów...')
    def import_source_documents(self, topic):
        docs = []
        files = []

        data_folder = os.path.join("data", topic)
        for file in os.listdir(data_folder):
            if file.endswith(".txt"):
                with open(os.path.join(data_folder, file), encoding='utf-8') as f:
                    docs.append(f.read())
                    files.append(file)

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        splits = []
        for i, doc in enumerate(docs):
            for chunk in text_splitter.split_text(doc):
                splits.append(Document(page_content=chunk, metadata={"source": files[i]}))

        vectordb = DocArrayInMemorySearch.from_documents(splits, self.embedding_model)

        retriever = vectordb.as_retriever(
            search_type='similarity',
            search_kwargs={'k': 2, 'fetch_k': 4}
        )

        memory = ConversationBufferMemory(
            memory_key='chat_history',
            output_key='answer',
            return_messages=True
        )

        system_message_prompt = SystemMessagePromptTemplate.from_template(
            """
            Jesteś chatbotem AI. Używaj informacji z poniższych dokumentów, aby dokładnie odpowiadać na pytania:
            {context}

            Odpowiedz na poniższe pytanie, korzystając wyłącznie z dokumentów:
            {question}
            """
        )

        prompt = ChatPromptTemplate.from_messages([system_message_prompt])

        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=retriever,
            memory=memory,
            return_source_documents=True,
            verbose=False,
            combine_docs_chain_kwargs={"prompt": prompt}
        )

        return qa_chain

    @utils.enable_chat_history
    def main(self):
        st.sidebar.header("Wybierz temat")
        topic = st.sidebar.selectbox("Tematy", ["f1", "sport", "nazwy"])

        user_query = st.chat_input(placeholder="Zadaj pytanie na temat " + topic)

        if user_query:
            qa_chain = self.import_source_documents(topic)

            utils.display_msg(user_query, 'user')

            with st.chat_message("assistant"):
                st_cb = StreamHandler(st.empty())

                result = qa_chain.invoke(
                    {"question": user_query},
                    {"callbacks": [st_cb]}
                )
                response = result["answer"]
                st.session_state.messages.append({"role": "assistant", "content": response})
                utils.print_qa(CustomDocChatbot, user_query, response)

                for doc in result['source_documents']:
                    filename = os.path.basename(doc.metadata['source'])
                    ref_title = f":blue[Dokument źródłowy: {filename}]"
                    with st.popover(ref_title):
                        st.caption(doc.page_content)

if __name__ == "__main__":
    obj = CustomDocChatbot()
    obj.main()
