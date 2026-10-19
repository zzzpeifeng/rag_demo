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


def split_into_chunks_sliding_window(doc_file, chunk_size=200, overlap=50) -> list[str]:
    """
        滑窗重叠切片（优先在标点处切分，避免切断句子）
    :param doc_file: 文档路径
    :param chunk_size: 每块的目标字数
    :param overlap: 相邻块的重叠字数
    :return: 切片列表
    """
    with open(doc_file, 'r') as file:
        content = file.read()

    if len(content) <= chunk_size:
        return [content]

    # 切分标点优先级：句末标点 > 换行 > 逗号/顿号
    sentence_end = '。！？!?；;\n'
    soft_punct = '，,、：:'

    def find_cut_pos(target: int) -> int:
        """在 target 附近寻找最近的标点位置，找不到则返回 target"""
        search_range = min(chunk_size // 2, target)
        # 先往前找句末标点
        for i in range(target, max(target - search_range, 0), -1):
            if content[i] in sentence_end:
                return i + 1
        # 再往前找软标点
        for i in range(target, max(target - search_range, 0), -1):
            if content[i] in soft_punct:
                return i + 1
        return target

    chunks = []
    start = 0
    while start < len(content):
        raw_end = min(start + chunk_size, len(content))
        if raw_end < len(content):
            end = find_cut_pos(raw_end)
        else:
            end = raw_end
        chunks.append(content[start:end])
        if end >= len(content):
            break
        start = max(end - overlap, start + 1)
    return chunks


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
    # 文档切片（滑窗重叠，优先在标点处切分）
    chunks = split_into_chunks_sliding_window('doc.md')
    for i, chunk in enumerate(chunks):
        print(f'[{i}] {chunk}')

    # 切片向量化
    embeddings = [embed_chunk(chunk) for chunk in chunks]

    # 拿到向量库的collection

    # 清空向量库旧数据
    existing = default_collection.get()
    if existing['ids']:
        default_collection.delete(ids=existing['ids'])

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
