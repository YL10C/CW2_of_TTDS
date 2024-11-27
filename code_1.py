# information_retrieval_evaluation.py

import pandas as pd
import numpy as np
from collections import Counter, defaultdict
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from gensim import corpora, models

class InformationRetrievalEvaluation:
    def __init__(self, system_results_file, qrels_file):
        self.system_results_file = system_results_file
        self.qrels_file = qrels_file
        self.system_results = None
        self.qrels = None
        self.qrels_dict = defaultdict(set)
        self.qrels_relevance = {}
        self.results = []

    def load_data(self):
        # 读取system_results.csv
        self.system_results = pd.read_csv(self.system_results_file)
        # 读取qrels.csv
        self.qrels = pd.read_csv(self.qrels_file)
        # 构建qrels字典
        for _, row in self.qrels.iterrows():
            qid = row['query_id']
            doc_id = row['doc_id']
            relevance = row['relevance']
            self.qrels_dict[qid].add(doc_id)
            self.qrels_relevance[(qid, doc_id)] = relevance  # 用于nDCG计算

    # 定义评估指标函数
    def precision_at_k(self, retrieved_docs, relevant_docs, k):
        retrieved_k = retrieved_docs[:k]
        num_relevant = sum([1 for doc in retrieved_k if doc in relevant_docs])
        return num_relevant / k

    def recall_at_k(self, retrieved_docs, relevant_docs, k):
        retrieved_k = retrieved_docs[:k]
        num_relevant = sum([1 for doc in retrieved_k if doc in relevant_docs])
        return num_relevant / len(relevant_docs) if len(relevant_docs) > 0 else 0

    def r_precision(self, retrieved_docs, relevant_docs):
        r = len(relevant_docs)
        retrieved_r = retrieved_docs[:r]
        num_relevant = sum([1 for doc in retrieved_r if doc in relevant_docs])
        return num_relevant / r if r > 0 else 0

    def average_precision(self, retrieved_docs, relevant_docs):
        num_relevant = 0
        sum_precisions = 0
        for idx, doc in enumerate(retrieved_docs):
            if doc in relevant_docs:
                num_relevant += 1
                precision = num_relevant / (idx + 1)
                sum_precisions += precision
        return sum_precisions / len(relevant_docs) if len(relevant_docs) > 0 else 0

    def dcg_at_k(self, retrieved_docs, qid, k):
        dcg = 0
        for i, doc in enumerate(retrieved_docs[:k]):
            relevance = self.qrels_relevance.get((qid, doc), 0)
            denom = np.log2(i + 2)  # i从0开始，所以需要+2
            dcg += relevance / denom
        return dcg

    def idcg_at_k(self, qid, k):
        relevances = [self.qrels_relevance[(qid, doc)] for doc in self.qrels_dict[qid]]
        relevances.sort(reverse=True)
        idcg = 0
        for i, relevance in enumerate(relevances[:k]):
            denom = np.log2(i + 2)
            idcg += relevance / denom
        return idcg

    def ndcg_at_k(self, retrieved_docs, qid, k):
        dcg = self.dcg_at_k(retrieved_docs, qid, k)
        idcg = self.idcg_at_k(qid, k)
        return dcg / idcg if idcg > 0 else 0

    def evaluate(self):
        # 初始化结果列表
        self.results = []
        # 获取所有系统编号
        systems = self.system_results['system_number'].unique()

        for system in systems:
            system_data = self.system_results[self.system_results['system_number'] == system]
            queries = system_data['query_number'].unique()
            # 初始化每个系统的指标列表
            system_metrics = {'P@10': [], 'R@50': [], 'r-precision': [], 'AP': [], 'nDCG@10': [], 'nDCG@20': []}

            for qid in queries:
                # 获取检索到的文档列表
                retrieved_docs = system_data[system_data['query_number'] == qid].sort_values('rank_of_doc')['doc_number'].tolist()
                # 获取相关文档集合
                relevant_docs = self.qrels_dict.get(qid, set())

                # 计算指标
                p10 = self.precision_at_k(retrieved_docs, relevant_docs, 10)
                r50 = self.recall_at_k(retrieved_docs, relevant_docs, 50)
                r_prec = self.r_precision(retrieved_docs, relevant_docs)
                ap = self.average_precision(retrieved_docs, relevant_docs)
                ndcg10 = self.ndcg_at_k(retrieved_docs, qid, 10)
                ndcg20 = self.ndcg_at_k(retrieved_docs, qid, 20)

                # 存储指标
                system_metrics['P@10'].append(p10)
                system_metrics['R@50'].append(r50)
                system_metrics['r-precision'].append(r_prec)
                system_metrics['AP'].append(ap)
                system_metrics['nDCG@10'].append(ndcg10)
                system_metrics['nDCG@20'].append(ndcg20)

                # 添加到结果列表
                self.results.append({
                    'system_number': system,
                    'query_number': qid,
                    'P@10': round(p10, 3),
                    'R@50': round(r50, 3),
                    'r-precision': round(r_prec, 3),
                    'AP': round(ap, 3),
                    'nDCG@10': round(ndcg10, 3),
                    'nDCG@20': round(ndcg20, 3)
                })

            # 计算平均值
            mean_p10 = np.mean(system_metrics['P@10'])
            mean_r50 = np.mean(system_metrics['R@50'])
            mean_r_prec = np.mean(system_metrics['r-precision'])
            mean_ap = np.mean(system_metrics['AP'])
            mean_ndcg10 = np.mean(system_metrics['nDCG@10'])
            mean_ndcg20 = np.mean(system_metrics['nDCG@20'])

            # 添加平均值到结果列表
            self.results.append({
                'system_number': system,
                'query_number': 'mean',
                'P@10': round(mean_p10, 3),
                'R@50': round(mean_r50, 3),
                'r-precision': round(mean_r_prec, 3),
                'AP': round(mean_ap, 3),
                'nDCG@10': round(mean_ndcg10, 3),
                'nDCG@20': round(mean_ndcg20, 3)
            })

    def save_results(self, output_file):
        # 转换结果为DataFrame
        results_df = pd.DataFrame(self.results)
        # 调整列的顺序
        results_df = results_df[['system_number', 'query_number', 'P@10', 'R@50', 'r-precision', 'AP', 'nDCG@10', 'nDCG@20']]
        # 保存为CSV文件
        results_df.to_csv(output_file, index=False)



# text_analysis.py
class TextAnalysis:
    def __init__(self, corpus_file):
        self.corpus_file = corpus_file
        self.data = []
        self.labels = []
        self.texts = []
        self.dictionary = None
        self.corpus = None
        self.lda_model = None
        self.num_topics = 20
        self.corpus_topic_averages = {}
        self.mi_scores = defaultdict(dict)
        self.chi2_scores = defaultdict(dict)

    def load_and_preprocess_data(self):
        # 确保已下载必要的nltk数据
        # nltk.download('punkt')
        # nltk.download('stopwords')

        # 初始化停用词列表和词干提取器
        self.stop_words = set(stopwords.words('english'))
        self.ps = PorterStemmer()

        # 读取数据并预处理
        with open(self.corpus_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    corpus_label, text = line.split('\t', 1)
                    tokens = self.preprocess_text(text)
                    self.data.append({'label': corpus_label, 'text': text, 'tokens': tokens})
                    self.labels.append(corpus_label)
                    self.texts.append(tokens)

    def preprocess_text(self, text):
        # 转小写
        text = text.lower()
        # 去除标点符号和特殊字符
        text = re.sub(r'[^a-z\s]', '', text)
        # 分词
        tokens = nltk.word_tokenize(text)
        # 去除停用词
        tokens = [word for word in tokens if word not in self.stop_words]
        # 词干提取
        tokens = [self.ps.stem(word) for word in tokens]
        return tokens

    def calculate_mi_chi2(self):
        # 初始化数据结构
        corpus_tokens = defaultdict(list)
        all_tokens = set()
        corpus_docs = defaultdict(int)
        total_docs = 0

        for item in self.data:
            label = item['label']
            tokens = set(item['tokens'])
            corpus_tokens[label].append(tokens)
            all_tokens.update(tokens)
            corpus_docs[label] += 1
            total_docs += 1

        # 获取所有类别
        labels_set = set(corpus_docs.keys())

        for label in labels_set:
            # 当前类别的文档数
            N_c = corpus_docs[label]
            # 其他类别的文档数
            N_not_c = total_docs - N_c

            # 统计当前类别中每个词的文档频率
            token_doc_freq_c = Counter()
            for tokens in corpus_tokens[label]:
                token_doc_freq_c.update(tokens)

            # 统计其他类别中每个词的文档频率
            token_doc_freq_not_c = Counter()
            for other_label in labels_set - {label}:
                for tokens in corpus_tokens[other_label]:
                    token_doc_freq_not_c.update(tokens)

            # 所有词的集合
            tokens = all_tokens

            for token in tokens:
                # 计算N11
                N11 = token_doc_freq_c.get(token, 0)
                # 计算N01
                N01 = token_doc_freq_not_c.get(token, 0)
                # 计算N10
                N10 = N_c - N11
                # 计算N00
                N00 = N_not_c - N01
                # 总文档数
                N = total_docs

                # 避免零值造成的数学错误，添加一个极小值
                epsilon = 1e-10

                # 计算 MI
                MI = 0
                # 计算每一项，如果频数为零，跳过该项
                # 第一项
                if N11 > 0:
                    MI += (N11 / N) * (np.log2((N * N11) / ((N11 + N10) * (N11 + N01) + epsilon)))
                # 第二项
                if N01 > 0:
                    MI += (N01 / N) * (np.log2((N * N01) / ((N01 + N00) * (N11 + N01) + epsilon)))
                # 第三项
                if N10 > 0:
                    MI += (N10 / N) * (np.log2((N * N10) / ((N11 + N10) * (N10 + N00) + epsilon)))
                # 第四项
                if N00 > 0:
                    MI += (N00 / N) * (np.log2((N * N00) / ((N01 + N00) * (N10 + N00) + epsilon)))

                self.mi_scores[label][token] = MI

                # 计算 χ²
                numerator = (N11 * N00 - N10 * N01) ** 2 * N
                denominator = (N11 + N01) * (N11 + N10) * (N10 + N00) * (N01 + N00) + epsilon
                chi2 = numerator / denominator
                self.chi2_scores[label][token] = chi2

    def save_mi_chi2_results(self):
        # 对每个类别的词按照得分排序
        for label in set(self.labels):
            # MI排序
            mi_sorted = sorted(self.mi_scores[label].items(), key=lambda x: x[1], reverse=True)
            # 保存MI结果
            with open(f'{label}_mi.txt', 'w', encoding='utf-8') as f:
                for token, score in mi_sorted:
                    f.write(f'{token},{score}\n')

            # χ²排序
            chi2_sorted = sorted(self.chi2_scores[label].items(), key=lambda x: x[1], reverse=True)
            # 保存χ²结果
            with open(f'{label}_chi2.txt', 'w', encoding='utf-8') as f:
                for token, score in chi2_sorted:
                    f.write(f'{token},{score}\n')

    def run_lda(self):
        # 创建词典
        self.dictionary = corpora.Dictionary(self.texts)
        # 创建语料库
        self.corpus = [self.dictionary.doc2bow(text) for text in self.texts]
        # 训练LDA模型
        self.lda_model = models.LdaModel(corpus=self.corpus, id2word=self.dictionary, num_topics=self.num_topics, passes=10, random_state=42)
        # 获取每个文档的主题分布
        doc_topics = self.lda_model.get_document_topics(self.corpus)
        # 初始化字典，存储每个语料库的主题分布列表
        corpus_topic_distributions = defaultdict(list)

        for idx, doc_topic in enumerate(doc_topics):
            label = self.labels[idx]
            # 将主题分布转换为数组
            topic_prob = np.zeros(self.num_topics)
            for topic_id, prob in doc_topic:
                topic_prob[topic_id] = prob
            corpus_topic_distributions[label].append(topic_prob)

        # 计算每个语料库的主题平均得分
        for label, topic_distributions in corpus_topic_distributions.items():
            # 将列表转换为矩阵
            topic_matrix = np.array(topic_distributions)
            # 计算每个主题的平均概率
            topic_avg = np.mean(topic_matrix, axis=0)
            self.corpus_topic_averages[label] = topic_avg

    def display_lda_results(self):
        for label, topic_avg in self.corpus_topic_averages.items():
            # 找到平均得分最高的主题编号
            top_topic = np.argmax(topic_avg)
            print(f"{label}语料库最相关的主题编号：{top_topic}")

            # 获取该主题的前10个词及其概率
            top_words = self.lda_model.show_topic(top_topic, topn=10)
            print(f"{label}语料库最相关主题的前10个词及其概率：")
            for word, prob in top_words:
                print(f"{word}: {prob}")
            print("\n")

        # 将主题平均得分转换为DataFrame
        topic_avg_df = pd.DataFrame(self.corpus_topic_averages, index=[f"Topic_{i}" for i in range(self.num_topics)])
        # 打印主题平均得分表
        print("各语料库的主题平均得分：")
        print(topic_avg_df)


if __name__ == '__main__':
    # 信息检索评估部分
    ir_eval = InformationRetrievalEvaluation(system_results_file='ttdssystemresults.csv', qrels_file='qrels.csv')
    ir_eval.load_data()
    ir_eval.evaluate()
    ir_eval.save_results(output_file='ir_eval.csv')

    # 文本分析部分
    text_analysis = TextAnalysis(corpus_file='bible_and_quran.tsv')
    text_analysis.load_and_preprocess_data()
    text_analysis.calculate_mi_chi2()
    text_analysis.save_mi_chi2_results()
    text_analysis.run_lda()
    text_analysis.display_lda_results()