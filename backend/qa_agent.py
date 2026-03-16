from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate


GOVERNING_PROMPT = """You are a document Q&A assistant. Use ONLY the context below to answer the question.
If the answer cannot be found in the context, respond with exactly:
"I cannot find this information in the provided document."

Context:
{context}

Question: {question}

Answer:"""


def build_agent(vectorstore, api_key: str):
    llm = ChatOpenAI(model_name="gpt-4.1", temperature=0, openai_api_key=api_key)
    retriever = vectorstore.as_retriever(search_type="similarity", k=4)

    prompt = PromptTemplate(
        template=GOVERNING_PROMPT,
        input_variables=["context", "question"]
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )

    return qa_chain, retriever
