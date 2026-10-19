import os

# 国内镜像加速 HuggingFace 模型下载
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from dotenv import load_dotenv
from sentence_transformers import CrossEncoder
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

load_dotenv()

# ============ 1. 文档加载 + 切片 ============
def load_and_split(doc_file: str) -> list:
    """
    加载文档并用 RecursiveCharacterTextSplitter 切片
    （LangChain 内置的递归切分器，优先在分隔符处切，自带 overlap）
    """
    loader = TextLoader(doc_file)
    docs = loader.load()

    # 中文友好的分隔符优先级：段落 > 换行 > 句末标点 > 软标点 > 空格 > 字符
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=50,
        separators=["\n\n", "\n", "。", "！", "？", "，", "、", "：", " ", ""],
    )
    return text_splitter.split_documents(docs)


# ============ 2. 向量化 + 存库 ============
# HuggingFaceEmbeddings 封装了 SentenceTransformer，不用手写 encode
embeddings = HuggingFaceEmbeddings(
    model_name="shibing624/text2vec-base-chinese",
    model_kwargs={"device": "cpu"},
)

# Chroma.from_documents 一步完成建库（自动清空重建）
vectorstore = Chroma(
    collection_name="default",
    embedding_function=embeddings,
    persist_directory="./chroma_langchain.db",
)


# ============ 3. 召回 ============
# retriever 封装了 query embedding + 向量检索
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})


# ============ 4. 重排（手动实现，保持与原版本一致的逻辑） ============
cross_encoder = CrossEncoder('cross-encoder/mmarco-mMiniLMv2-L12-H384-v1')
RELEVANCE_THRESHOLD = 0.0


def rerank(query: str, documents, top_k: int = 3):
    """用 cross-encoder 重排，返回 top_k 片段和最高分"""
    pairs = [(query, doc.page_content) for doc in documents]
    scores = cross_encoder.predict(pairs)

    doc_with_score = [(doc, score) for doc, score in zip(documents, scores)]
    doc_with_score.sort(key=lambda x: x[1], reverse=True)

    top_docs = [doc for doc, _ in doc_with_score][:top_k]
    top_score = doc_with_score[0][1] if doc_with_score else -float('inf')
    return top_docs, top_score


# ============ 5. 生成（LCEL 链式编排） ============
llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
    temperature=0.7,
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个问答助手，根据用户提供的参考内容回答问题。参考内容没有的信息不要编造。"),
    ("human", """参考内容：
{context}

问题：{query}

请根据参考内容回答问题。如果参考内容中没有相关信息，请回答"根据现有资料无法回答"。"""),
])


def format_context(documents) -> str:
    """把重排后的文档列表格式化成 prompt 的 context 字符串"""
    return "\n\n".join(f"[{i}] {doc.page_content}" for i, doc in enumerate(documents))


# LCEL 链：query -> 召回 -> 重排 -> 拼上下文 -> prompt -> llm -> 字符串
rag_chain = (
    {
        "query": RunnablePassthrough(),
        "context": (lambda q: retrieve_and_rerank(q)) | RunnableLambda(format_context),
    }
    | prompt
    | llm
    | StrOutputParser()
)


def retrieve_and_rerank(query: str):
    """召回 + 重排，带阈值过滤"""
    retrieved = retriever.invoke(query)
    top_docs, top_score = rerank(query, retrieved, top_k=3)

    print(f'最高相关性分数: {top_score:.4f}（阈值: {RELEVANCE_THRESHOLD}）\n')
    for i, doc in enumerate(top_docs):
        print(f'[{i}] {doc.page_content}')

    if top_score < RELEVANCE_THRESHOLD:
        return None
    return top_docs


# ============ 6. 主流程 ============
def before():
    """建库：加载文档 -> 切片 -> 向量化 -> 存入 Chroma"""
    print('加载并切片文档...')
    docs = load_and_split('doc.md')
    print(f'共切出 {len(docs)} 块\n')

    # 清空旧数据
    existing = vectorstore.get()
    if existing['ids']:
        vectorstore.delete(ids=existing['ids'])

    # 存入向量库（Chroma 自动做向量化，不用手动 encode）
    vectorstore.add_documents(docs)
    print('已存入向量库')


def after():
    """问答：召回 -> 重排 -> 生成"""
    test_query = '多啦A梦使用的三个道具是什么？'

    print('召回\n')
    retrieved = retriever.invoke(test_query)
    for i, doc in enumerate(retrieved):
        print(f'{i}--{doc.page_content}')

    print('\n重排\n')
    top_docs, top_score = rerank(test_query, retrieved, top_k=3)
    print(f'最高相关性分数: {top_score:.4f}（阈值: {RELEVANCE_THRESHOLD}）\n')
    for i, doc in enumerate(top_docs):
        print(f'[{i}] {doc.page_content}')

    if top_score < RELEVANCE_THRESHOLD:
        print('\n生成\n')
        print('知识库中没有与问题相关的内容，无法回答。')
    else:
        print('\n生成\n')
        # 直接调用 LCEL 链
        answer = rag_chain.invoke(test_query)
        print(answer)


if __name__ == '__main__':
    # before()
    after()
