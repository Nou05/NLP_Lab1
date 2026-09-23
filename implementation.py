# ==============================================================================
# AI contribution:
# - Generated initial implementation of count vectorizer, TF, IDF, TF-IDF, and cosine similarity functions.
# - Suggested handling division by zero for document normalization and IDF computing.
# - I modified the implementation and verified it against small corpus unit tests.
# ==============================================================================

import math
import numpy as np

def build_vocabulary(corpus):
    """Xây dựng danh sách các từ duy nhất (Vocabulary) được sắp xếp theo alphabet."""
    vocab = set()
    for doc in corpus:
        tokens = doc.lower().split()
        vocab.update(tokens)
    return sorted(list(vocab))

def compute_counts(corpus, vocab):
    """Tính Count Vector cho từng document."""
    counts = []
    for doc in corpus:
        tokens = doc.lower().split()
        doc_counts = [tokens.count(term) for term in vocab]
        counts.append(doc_counts)
    return np.array(counts)

def compute_tf(counts):
    """Tính Term Frequency (TF): tf(t, d) = count(t, d) / sum(count(t', d))."""
    sum_terms_per_doc = counts.sum(axis=1, keepdims=True)
    # Tránh lỗi chia cho 0 nếu document rỗng
    sum_terms_per_doc[sum_terms_per_doc == 0] = 1
    return counts / sum_terms_per_doc

def compute_idf(corpus, vocab):
    """Tính Inverse Document Frequency (IDF): idf(t) = log(N / df(t))."""
    N = len(corpus)
    idf = {}
    for term in vocab:
        # Đếm số lượng document chứa term
        df = sum(1 for doc in corpus if term in doc.lower().split())
        # Công thức chuẩn trong slide Lab: idf = log(N / df)
        idf[term] = math.log(N / df) if df > 0 else 0.0
    return idf

def compute_tfidf(tf, idf, vocab):
    """Tính ma trận TF-IDF = TF * IDF."""
    idf_vector = np.array([idf[term] for term in vocab])
    return tf * idf_vector

def cosine_similarity(vec1, vec2):
    """Tính Cosine Similarity giữa 2 vectors: cos(x, y) = (x . y) / (||x|| * ||y||)."""
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    
    if norm_vec1 == 0 or norm_vec2 == 0:
        return 0.0
    return dot_product / (norm_vec1 * norm_vec2)


# ==============================================================================
# Unit Tests (Kiểm thử trên Corpus nhỏ ở Phần 8.3 của Lab)
# ==============================================================================
if __name__ == "__main__":
    # Corpus kiểm thử nhỏ
    test_corpus = [
        "cat eats fish",
        "dog eats fish",
        "cat likes fish"
    ]
    
    # 1. Test build_vocabulary
    vocab = build_vocabulary(test_corpus)
    print("Vocabulary:", vocab)
    assert vocab == ['cat', 'dog', 'eats', 'fish', 'likes'], "Vocabulary Test Failed!"
    
    # 2. Test compute_counts
    counts = compute_counts(test_corpus, vocab)
    print("Counts D1:", counts[0])
    
    # 3. Test compute_tf
    tf = compute_tf(counts)
    # tf('cat', D1) = 1/3 ~ 0.3333333
    assert abs(tf[0][vocab.index('cat')] - (1/3)) < 1e-6, "TF Test Failed!"
    
    # 4. Test compute_idf
    idf = compute_idf(test_corpus, vocab)
    # df('fish') = 3 -> idf('fish') = log(3/3) = 0.0
    assert abs(idf['fish'] - 0.0) < 1e-6, "IDF Test Failed!"
    
    # 5. Test compute_tfidf
    tfidf = compute_tfidf(tf, idf, vocab)
    
    # 6. Test cosine_similarity
    sim_1_2 = cosine_similarity(tfidf[0], tfidf[1])
    print(f"Cosine similarity (D1, D2): {sim_1_2:.4f}")
    
    print("\n✅ TẤT CẢ CÁC UNIT TESTS ĐÃ PASS RẤT TỐT!")