import os

# 国内镜像加速 HuggingFace 模型下载
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from dotenv import load_dotenv
from sentence_transformers import CrossEncoder
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnableBranch

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

# 构造器只负责打开/创建 collection，不做任何计算；向量化发生在 add_documents 时
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
    # 把「问题 + 每个候选片段」拼成一组 pair；cross-encoder 必须成对输入才能判断相关性（bi-encoder 才是各自独立编码）
    pairs = [(query, doc.page_content) for doc in documents]
    # 批量打分：一次推理搞定所有 pair，返回 numpy 数组，元素顺序与 pairs 严格一一对应
    scores = cross_encoder.predict(pairs)

    # 把「文档对象」和「它的分数」绑成元组，为排序做准备（Document 本身不携带分数）
    doc_with_score = [(doc, score) for doc, score in zip(documents, scores)]
    # 按分数从高到低排序（reverse=True），排完后下标 0 就是最相关的片段
    doc_with_score.sort(key=lambda x: x[1], reverse=True)

    # 截取前 top_k 个文档对象，丢弃分数（生成环节只需要正文文本）
    top_docs = [doc for doc, _ in doc_with_score][:top_k]
    # 取最高分用于阈值判断；若候选列表为空则返回 -inf，保证后续「分数 < 阈值」必然成立、自动走拒答分支
    top_score = doc_with_score[0][1] if doc_with_score else -float('inf')
    # 返回重排结果和最高分：调用方靠 top_score 决定是继续调大模型还是直接拒答
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


def retrieve_and_rerank(query: str) -> dict:
    """召回 + 重排，带阈值过滤；返回 {'query': 原问题, 'docs': 文档列表 或 None}"""
    retrieved = retriever.invoke(query)

    print('召回\n')
    for i, doc in enumerate(retrieved):
        print(f'{i}--{doc.page_content}')

    top_docs, top_score = rerank(query, retrieved, top_k=3)

    print('\n重排\n')
    print(f'最高相关性分数: {top_score:.4f}（阈值: {RELEVANCE_THRESHOLD}）\n')
    for i, doc in enumerate(top_docs):
        print(f'[{i}] {doc.page_content}')

    if top_score < RELEVANCE_THRESHOLD:
        return {'query': query, 'docs': None}
    return {'query': query, 'docs': top_docs}


def build_prompt_input(state: dict) -> dict:
    """把 {query, docs} 转成 prompt 需要的变量表 {query, context}"""
    return {'query': state['query'], 'context': format_context(state['docs'])}


REFUSAL_ANSWER = '知识库中没有与问题相关的内容，无法回答。'

# LCEL 链：query -> 召回+重排 -> [拒答 | 拼上下文 -> prompt -> llm -> 字符串]
# - 整条链只检索一次，中间产物由 retrieve_and_rerank 打印
# - 阈值不达标时走拒答分支，不调用大模型（把 if 挪进链里，避免 format_context(None) 崩溃）
rag_chain = (
    RunnableLambda(retrieve_and_rerank)
    | RunnableBranch(
        (lambda state: state['docs'] is None, RunnableLambda(lambda _: REFUSAL_ANSWER)),
        RunnableLambda(build_prompt_input) | prompt | llm | StrOutputParser(),
    )
)


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
    """问答：召回 -> 重排 -> 生成（整条链只执行一次）"""
    test_query = '多啦A梦使用的三个道具是什么？'

    answer = rag_chain.invoke(test_query)  # 召回/重排的打印在链内部完成
    print('\n生成\n')
    print(answer)


if __name__ == '__main__':
    before()
    after()
