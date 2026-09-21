"""随机森林基线：TF-IDF 特征 + 随机森林分类器。

传统机器学习路线，作为深度模型的对照基线：
训练快、可解释、能快速暴露数据本身的问题，为后续模型提供性能下界参考。
文本经 jieba 分词并过滤停用词后，用 TF-IDF 转稀疏向量再训练随机森林。

用法：
    python -m src.rf_baseline                        # 全量训练
    python -m src.rf_baseline --subsample 20000      # 快速验证（分钟级）
"""
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
from typing import List

import argparse
import json
import logging
import pickle
import time

import jieba

from src.config import CKPT_DIR, DEV_FILE, SEED, TEST_FILE, TRAIN_FILE
from src.data import load_txt, subsample

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

STOPWORDS_FILE = Path(__file__).resolve().parent.parent / "data" / "stopwords.txt"


def load_stopwords() -> set:
    """读取停用词表（749 个），供分词后过滤。"""
    return {w.strip() for w in STOPWORDS_FILE.read_text(encoding="utf-8").splitlines()
            if w.strip()}


def cut_texts(texts: List[str], stopwords: set) -> List[str]:
    """jieba 分词并过滤停用词，返回空格分隔的词串（TF-IDF 直接可用）。"""
    return [" ".join(w for w in jieba.lcut(t) if w.strip() and w not in stopwords)
            for t in texts]


def main(args) -> None:
    stopwords = load_stopwords()
    train_t, train_l = load_txt(TRAIN_FILE)
    dev_t, dev_l = load_txt(DEV_FILE)
    test_t, test_l = load_txt(TEST_FILE)
    if args.subsample:
        train_t, train_l = subsample(train_t, train_l, args.subsample, SEED)
    logger.info(f"train={len(train_t)}，开始分词（约 1 万条/分钟）")

    t0 = time.time()
    x_train = cut_texts(train_t, stopwords)
    x_dev = cut_texts(dev_t, stopwords)
    x_test = cut_texts(test_t, stopwords)
    logger.info(f"分词完成，用时 {time.time() - t0:.0f}s")

    tfidf = TfidfVectorizer(tokenizer=str.split, stop_words=None, lowercase=False,
                            token_pattern=None, max_features=args.max_features)
    x_train_vec = tfidf.fit_transform(x_train)
    x_dev_vec = tfidf.transform(x_dev)
    x_test_vec = tfidf.transform(x_test)
    logger.info(f"TF-IDF 特征维度: {x_train_vec.shape}")

    model = RandomForestClassifier(n_estimators=args.n_estimators, n_jobs=-1,
                                   random_state=SEED)
    t0 = time.time()
    model.fit(x_train_vec, train_l)
    logger.info(f"随机森林训练完成，用时 {time.time() - t0:.0f}s")

    dev_acc = accuracy_score(dev_l, model.predict(x_dev_vec))
    test_pred = model.predict(x_test_vec)
    test_acc = accuracy_score(test_l, test_pred)
    test_f1 = f1_score(test_l, test_pred, average="macro")
    logger.info(f"dev_acc={dev_acc:.4f} test_acc={test_acc:.4f} macro_f1={test_f1:.4f}")

    # 单条推理延迟（含分词与特征转换）
    sample = "今天下午的会议改到几点了"
    tfidf.transform(cut_texts([sample], stopwords))  # 预热
    t0 = time.perf_counter()
    for _ in range(20):
        model.predict(tfidf.transform(cut_texts([sample], stopwords)))
    latency = (time.perf_counter() - t0) / 20 * 1000

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "tfidf.pkl", "wb") as f:
        pickle.dump(tfidf, f)
    with open(out_dir / "rf.pkl", "wb") as f:
        pickle.dump(model, f)
    (out_dir / "metrics.json").write_text(json.dumps({
        "args": vars(args),
        "dev_acc": dev_acc, "test_acc": test_acc, "macro_f1": test_f1,
        "latency_ms": latency,
        "model_file_mb": (out_dir / "rf.pkl").stat().st_size / 1024 / 1024,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"模型已保存到 {out_dir}，单条延迟 {latency:.1f}ms")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="TF-IDF + 随机森林基线")
    ap.add_argument("--subsample", type=int, default=0, help="训练子采样条数，0 为全量")
    ap.add_argument("--n-estimators", type=int, default=100, help="随机森林树数量")
    ap.add_argument("--max-features", type=int, default=50000,
                    help="TF-IDF 最大特征数，限制内存与模型体积")
    ap.add_argument("--output-dir", default=str(CKPT_DIR / "rf"))
    main(ap.parse_args())
