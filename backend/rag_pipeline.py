from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document


def chunk_article(article_text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
    chunks = splitter.split_text(article_text)
    return [Document(page_content=chunk) for chunk in chunks]

def create_vectorstore(chunks):
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embedding_model)
    return vectorstore


if __name__ == "__main__":
    from data_loader import get_sample_paper

    article, _ = get_sample_paper()
    chunks = chunk_article(article)
    print(f"\n✅ Split into {len(chunks)} chunks.")

    vectorstore = create_vectorstore(chunks)
    retriever = vectorstore.as_retriever()
    results = retriever.get_relevant_documents("What problem does the paper address?")

    print("\n🔍 Top matching chunk:\n", results[0].page_content[:500])
