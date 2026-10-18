import os

# 国内镜像加速 HuggingFace 模型下载（必须在导入 sentence_transformers 之前设置）
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from dotenv import load_dotenv
from chromadb.types import Collection
from sentence_transformers import SentenceTransformer
import chromadb
from sentence_transformers import CrossEncoder
from openai import OpenAI

load_dotenv()

embdding_model = SentenceTransformer('shibing624/text2vec-base-chinese')
chromadb_client = chromadb.PersistentClient('./chroma.db')
default_collection = chromadb_client.get_or_create_collection(name='default')

llm_client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)


def split_into_chunks(doc_file) -> list[str]:
    """
     切片
    :param doc_file:
    :return:
    """
    with open(doc_file, 'r') as file:
        content = file.read()
    return [chunk for chunk in content.split('\n\n')]


def embed_chunk(chunk: str) -> list[float]:
    """
        将切片向量化
    :param chunk:
    :return:
    """
    embedding = embdding_model.encode(chunk)
    return embedding.tolist()


def save_embeddings(chunks: list[str], embeddings: list[list[float]]) -> None:
    """
        将数据存入向量库
    :param chunks:
    :param embeddings:
    :param chromadb_collection:
    :return:
    """
    ids = [str(i) for i in range(len(chunks))]
    default_collection.add(
        documents=chunks,
        embeddings=embeddings,
        ids=ids
    )


def retrieve(query: str, top_k: int) -> list[str]:
    """
        召回
        从向量库中获取与query最相关的top_k个片段
    :param query:
    :param top_k:
    :return:
    """
    query_embedding = embed_chunk(query)  # 用户输入的问题
    results = default_collection.query(
        query_embeddings=query_embedding,
        n_results=top_k
    )
    return results['documents'][0]


def rerank(query: str, retrieved_chunks: list[str], top_k: int) -> tuple[list[str], float]:
    """
        重排
        返回重排后的 top_k 片段，以及最高分（用于相关性判断）
    """
    cross_encoder = CrossEncoder('cross-encoder/mmarco-mMiniLMv2-L12-H384-v1')
    pairs = [(query, chunk) for chunk in retrieved_chunks]
    scores = cross_encoder.predict(pairs)

    chunk_with_score_list = [(chunk, score) for chunk, score in zip(retrieved_chunks, scores)]

    chunk_with_score_list.sort(key=lambda pair: pair[1], reverse=True)
    top_chunks = [chunk for chunk, _ in chunk_with_score_list][:top_k]
    top_score = chunk_with_score_list[0][1] if chunk_with_score_list else -float('inf')
    return top_chunks, top_score


def generate(query: str, context_chunks: list[str]) -> str:
    """
        生成
        将重排后的上下文片段与用户问题组装成 prompt，调用大模型生成回答
    :param query: 用户问题
    :param context_chunks: 重排后的相关片段
    :return: 大模型生成的回答
    """
    context = "\n\n".join(f"[{i}] {chunk}" for i, chunk in enumerate(context_chunks))
    prompt = f"""参考内容：
{context}

问题：{query}

请根据参考内容回答问题。如果参考内容中没有相关信息，请回答"根据现有资料无法回答"。"""

    response = llm_client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "你是一个问答助手，根据用户提供的参考内容回答问题。参考内容没有的信息不要编造。"},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7
    )
    return response.choices[0].message.content


def before():
    # 文档切片
    chunks = split_into_chunks('doc.md')
    for i, chunk in enumerate(chunks):
        print(f'[{i}] {chunk}')

    # 切片向量化
    embeddings = [embed_chunk(chunk) for chunk in chunks]

    # 拿到向量库的collection

    # 存入向量库
    save_embeddings(chunks, embeddings)


def after():
    test_query = '多啦A梦使用的三个道具是什么？'

    # 召回，根据输入内容，在向量库中找到最匹配的5个切片
    retrieved_chunks = retrieve(test_query, 5)
    print('召回\n')
    for i, chunk in enumerate(retrieved_chunks):
        print(f'{i}--{chunk}')

    # 重排
    print('重排\n')
    RELEVANCE_THRESHOLD = 0.0  # cross-encoder 相关性阈值，低于此值认为无相关内容
    rerank_chunks, top_score = rerank(test_query, retrieved_chunks, 3)
    print(f'最高相关性分数: {top_score:.4f}（阈值: {RELEVANCE_THRESHOLD}）\n')

    for i, chunk in enumerate(rerank_chunks):
        print(f'[{i}] {chunk}')

    # 阈值过滤：无相关内容时直接拒答，不调用大模型
    if top_score < RELEVANCE_THRESHOLD:
        print('\n生成\n')
        print('知识库中没有与问题相关的内容，无法回答。')
    else:
        # 生成
        print('\n生成\n')
        answer = generate(test_query, rerank_chunks)
        print(answer)


if __name__ == '__main__':
    # before()
    after()
