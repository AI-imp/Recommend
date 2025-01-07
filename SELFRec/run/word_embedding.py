import jieba
from gensim.models import Word2Vec, KeyedVectors
from numpy import dot
from numpy.linalg import norm
import numpy as np


class word2vec_process:
    def __init__(self, model_dir):
        self.model = KeyedVectors.load_word2vec_format(model_dir, binary=True)

    def sentence_vector(self, sentence):
        model = self.model

        words = jieba.lcut(sentence)
        # 分词
        # words = jieba.lcut(sentence)

        return sum([model[word] for word in words if word in model]) / len(words)

    def words_vector(self, words):
        return self.model[words]

    # 分词并计算句子与词汇的相似度
    def calculate_similarity(self, sentence_vector, word_vector):
        # 计算余弦相似度
        similarity = self.cosine_similarity(sentence_vector, word_vector)
        return similarity

    def cosine_similarity(self, v1, v2):

        cos_sim = dot(v1, v2) / (norm(v1) * norm(v2))
        return cos_sim

    def topk_relation(self, sentence, words, k):
        sentence_vec = self.sentence_vector(sentence)
        similarity = []
        for word in words:
            word_vec = self.sentence_vector(word)
            # 表示没有当前词向量嵌入
            if type(self.sentence_vector(word)) == float:
                # print(word)
                if word in sentence:
                    similarity.append(1)
                else:
                    similarity.append(0)
            else:
                similarity.append(self.cosine_similarity(sentence_vec, word_vec))

        arr = np.array(similarity)
        sorted_indices = np.argsort(-arr)
        topk_indices = [words[lable] for lable in sorted_indices[:k]]
        return topk_indices


if __name__ == '__main__':
    vec = word2vec_process(r'C:\Users\12563\Desktop\coda\SELFRec\setting\baike_26g_news_13g_novel_229g.bin')
