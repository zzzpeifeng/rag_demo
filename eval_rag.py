import os

# 国内镜像加速 HuggingFace 模型下载（必须在导入 sentence_transformers 之前设置）
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from sentence_transformers import SentenceTransformer, CrossEncoder

from main import split_into_chunks_sliding_window, embed_chunk
import numpy as np

embdding_model = SentenceTransformer('shibing624/text2vec-base-chinese')

# 可选重排器：None=不重排（召回直接取 top-3）；也可指定其他模型做对照
# RERANKER = None
RERANKER = "BAAI/bge-reranker-base"

cross_encoder = CrossEncoder(RERANKER) if RERANKER else None

# ===================== 15 道测试题 =====================
# gold_anchors: 答案所在原文的唯一子串（跨切片策略定位）
# 空列表 = 库外题，应当拒答
CASES = [
    # ---- 事实题（8 道）----
    {"id": "F1", "type": "fact", "question": "哆啦A梦决定使用的三件秘密道具分别是什么？",
     "gold_anchors": ["三件秘密道具分别是"]},
    {"id": "F2", "type": "fact", "question": "黑暗赛亚人是怎么被制造出来的？",
     "gold_anchors": ["复制了贝吉塔的基因"]},
    {"id": "F3", "type": "fact", "question": "特兰克斯为什么认为哆啦A梦能帮上忙？",
     "gold_anchors": ["唯一缺失的武器"]},
    {"id": "F4", "type": "fact", "question": "大雄在精神屋里实际经历了多长时间？",
     "gold_anchors": ["整整一年的苦修"]},
    {"id": "F5", "type": "fact", "question": "给黑暗赛亚人最后一击的是谁、用了什么方式？",
     "gold_anchors": ["击穿了黑暗赛亚人的能量核心"]},
    {"id": "F6", "type": "fact", "question": "哆啦A梦他们是怎样到达未来世界的？",
     "gold_anchors": ["启动时光机"]},
    {"id": "F7", "type": "fact", "question": "战斗结束后，大雄回到现代有哪些改变？",
     "gold_anchors": ["不再轻易抱怨"]},
    {"id": "F8", "type": "fact", "question": "结尾电视新闻里报道了什么？",
     "gold_anchors": ["击退了失控的机器人"]},
    # ---- 跨块题（3 道）----
    {"id": "C1", "type": "multi", "question": "大雄最开始是个什么样的人，后来又变成了什么样？",
     "gold_anchors": ["连第一页都没动", "不再轻易抱怨"]},
    {"id": "C2", "type": "multi", "question": "击败黑暗赛亚人的过程中，三件道具各自起了什么作用？",
     "gold_anchors": ["复制斗篷", "时间停止手表", "击穿了黑暗赛亚人的能量核心"]},
    {"id": "C3", "type": "multi", "question": "为什么文中说\"这一次是大雄救了世界\"？",
     "gold_anchors": ["仅凭特兰克斯一人根本无法压制", "是大雄救了世界"]},
    # ---- 库外题（4 道，gold 为空 = 应拒答）----
    {"id": "N1", "type": "neg_easy", "question": "红烧肉怎么做才好吃？", "gold_anchors": []},
    {"id": "N2", "type": "neg_easy", "question": "Python 的 GIL 是什么？", "gold_anchors": []},
    {"id": "N3", "type": "neg_hard", "question": "贝吉塔第一次登场是在哪一集？", "gold_anchors": []},
    {"id": "N4", "type": "neg_hard", "question": "静香最后和谁在一起了？", "gold_anchors": []},
]

RELEVANCE_THRESHOLD = 0.1


def retrieve_chunks(query: str, chunks: list[str], chunk_embeddings: np.ndarray, top_k: int) -> list[int]:
    """召回：余弦相似度取 top_k，返回 chunk 下标列表"""
    q = embed_chunk(query)
    q = np.array(q)
    sims = chunk_embeddings @ q / (np.linalg.norm(chunk_embeddings, axis=1) * np.linalg.norm(q) + 1e-9)
    top_idx = np.argsort(sims)[::-1][:top_k]
    return top_idx.tolist()


def gold_rank(case: dict, chunks_in_order: list[str]) -> int | None:
    """在有序块列表中找到 gold 锚点最早出现的位置（1-based），未命中返回 None"""
    anchors = case["gold_anchors"]
    if not anchors:
        return None
    for rank, chunk in enumerate(chunks_in_order, start=1):
        if any(a in chunk for a in anchors):
            return rank
    return None


def main():
    chunks = split_into_chunks_sliding_window('doc.md')
    chunk_embeddings = np.array([embed_chunk(c) for c in chunks])
    print(f"知识库共 {len(chunks)} 块\n")

    # 每题一行记录
    rows = []
    pos_scores, neg_scores = [], []

    for case in CASES:
        q = case["question"]
        # 召回 top5
        ret_idx = retrieve_chunks(q, chunks, chunk_embeddings, 5)
        ret_chunks = [chunks[i] for i in ret_idx]
        # 重排 top3（RERANKER=None 时走"无重排基线"：召回直接取前3，分数用 rank1 的 cosine）
        if cross_encoder is None:
            rerank_chunks = ret_chunks[:3]
            qv = np.array(embed_chunk(q))
            top_score = float(chunk_embeddings[ret_idx[0]] @ qv /
                              (np.linalg.norm(chunk_embeddings[ret_idx[0]]) * np.linalg.norm(qv) + 1e-9))
        else:
            pairs = [(q, c) for c in ret_chunks]
            scores = cross_encoder.predict(pairs)
            order = np.argsort(scores)[::-1]
            rerank_chunks = [ret_chunks[i] for i in order[:3]]
            top_score = float(scores[order[0]])

        rec = {"id": case["id"], "type": case["type"], "question": q}
        rec["top_score"] = float(top_score)

        if case["gold_anchors"]:
            # 正样本
            rec["ret_rank"] = gold_rank(case, ret_chunks)       # 召回阶段 rank
            rec["rerank_rank"] = gold_rank(case, rerank_chunks) # 重排阶段 rank
            rec["refused"] = top_score < RELEVANCE_THRESHOLD
            pos_scores.append(top_score)
        else:
            # 负样本
            rec["refused"] = top_score < RELEVANCE_THRESHOLD
            neg_scores.append(top_score)

        rows.append(rec)

    # ===================== 汇总指标 =====================
    pos = [r for r in rows if r["type"] in ("fact", "multi")]
    neg = [r for r in rows if r["type"].startswith("neg")]

    hit5 = sum(1 for r in pos if r.get("ret_rank") is not None) / len(pos)
    hit3 = sum(1 for r in pos if r.get("rerank_rank") is not None) / len(pos)
    mrr5 = sum(1 / r["ret_rank"] for r in pos if r.get("ret_rank")) / len(pos)
    mrr3 = sum(1 / r["rerank_rank"] for r in pos if r.get("rerank_rank")) / len(pos)

    false_reject = sum(1 for r in pos if r["refused"]) / len(pos)   # 误拒率
    false_accept = sum(1 for r in neg if not r["refused"]) / len(neg)  # 漏拒率

    print("=" * 60)
    print("【检索层】")
    print(f"  Hit@5  (召回命中率): {hit5:.3f}  ({sum(1 for r in pos if r.get('ret_rank') is not None)}/{len(pos)})")
    print(f"  Hit@3  (重排命中率): {hit3:.3f}  ({sum(1 for r in pos if r.get('rerank_rank') is not None)}/{len(pos)})")
    print(f"  MRR@5  (召回倒数排名): {mrr5:.3f}")
    print(f"  MRR@3  (重排倒数排名): {mrr3:.3f}")

    print("\n【拒答层】")
    print(f"  误拒率 (有答案被拒): {false_reject:.3f}  ({sum(1 for r in pos if r['refused'])}/{len(pos)})")
    print(f"  漏拒率 (库外被硬答): {false_accept:.3f}  ({sum(1 for r in neg if not r['refused'])}/{len(neg)})")

    print("\n【分数分布（阈值标定依据）】")
    print(f"  正样本 top_score: {sorted(round(s, 2) for s in pos_scores)}")
    print(f"  负样本 top_score: {sorted(round(s, 2) for s in neg_scores)}")

    # 明细
    print("\n【逐题明细】")
    for r in rows:
        if r["type"] in ("fact", "multi"):
            rr = r.get("ret_rank") or "-"
            rk = r.get("rerank_rank") or "-"
            flag = "误拒" if r["refused"] else "OK"
            print(f"  {r['id']:>3} | 召回rank={rr} 重排rank={rk} score={r['top_score']:+.2f} {flag} | {r['question'][:22]}")
        else:
            flag = "漏拒!" if not r["refused"] else "拒答OK"
            print(f"  {r['id']:>3} | score={r['top_score']:+.2f} {flag} | {r['question'][:22]}")


if __name__ == '__main__':
    main()
