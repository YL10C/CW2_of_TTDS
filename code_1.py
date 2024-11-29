# information_retrieval_evaluation.py

import csv
import pandas as pd
import numpy as np
from collections import Counter, defaultdict
import nltk
import re
from nltk.corpus import stopwords
from gensim import corpora, models
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.calibration import LinearSVC
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn import svm
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, precision_recall_fscore_support, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, train_test_split
from scipy.sparse import hstack
from scipy.stats import ttest_rel

# Download necessary NLTK data
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')


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
        # Read system_results.csv
        self.system_results = pd.read_csv(self.system_results_file)
        # Read qrels.csv
        self.qrels = pd.read_csv(self.qrels_file)
        # Build qrels dictionary
        for _, row in self.qrels.iterrows():
            qid = row['query_id']
            doc_id = row['doc_id']
            relevance = row['relevance']
            self.qrels_dict[qid].add(doc_id)
            self.qrels_relevance[(qid, doc_id)] = relevance  # For nDCG calculation

    # Define evaluation metric functions
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
            denom = np.log2(i + 2)  # i starts from 0, so add 2
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
        # Initialize results list
        self.results = []
        # Get all system numbers
        systems = self.system_results['system_number'].unique()

        for system in systems:
            system_data = self.system_results[self.system_results['system_number'] == system]
            queries = system_data['query_number'].unique()
            # Initialize metrics list for each system
            system_metrics = {'P@10': [], 'R@50': [], 'r-precision': [], 'AP': [], 'nDCG@10': [], 'nDCG@20': []}

            for qid in queries:
                # Get the list of retrieved documents
                retrieved_docs = system_data[system_data['query_number'] == qid].sort_values('rank_of_doc')['doc_number'].tolist()
                # Get the set of relevant documents
                relevant_docs = self.qrels_dict.get(qid, set())

                # Compute metrics
                p10 = self.precision_at_k(retrieved_docs, relevant_docs, 10)
                r50 = self.recall_at_k(retrieved_docs, relevant_docs, 50)
                r_prec = self.r_precision(retrieved_docs, relevant_docs)
                ap = self.average_precision(retrieved_docs, relevant_docs)
                ndcg10 = self.ndcg_at_k(retrieved_docs, qid, 10)
                ndcg20 = self.ndcg_at_k(retrieved_docs, qid, 20)

                # Store metrics
                system_metrics['P@10'].append(p10)
                system_metrics['R@50'].append(r50)
                system_metrics['r-precision'].append(r_prec)
                system_metrics['AP'].append(ap)
                system_metrics['nDCG@10'].append(ndcg10)
                system_metrics['nDCG@20'].append(ndcg20)

                # Add to results list
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

            # Compute averages
            mean_p10 = np.mean(system_metrics['P@10'])
            mean_r50 = np.mean(system_metrics['R@50'])
            mean_r_prec = np.mean(system_metrics['r-precision'])
            mean_ap = np.mean(system_metrics['AP'])
            mean_ndcg10 = np.mean(system_metrics['nDCG@10'])
            mean_ndcg20 = np.mean(system_metrics['nDCG@20'])

            # Add averages to results list
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
        # Convert results to DataFrame
        results_df = pd.DataFrame(self.results)
        # Rearrange columns
        results_df = results_df[['system_number', 'query_number', 'P@10', 'R@50', 'r-precision', 'AP', 'nDCG@10', 'nDCG@20']]
        # Save to CSV file
        results_df.to_csv(output_file, index=False)

    def perform_t_tests(self):
        # Convert results to DataFrame
        results_df = pd.DataFrame(self.results)
        # Exclude 'mean' rows
        results_df = results_df[results_df['query_number'] != 'mean']
        
        # Get list of systems
        systems = results_df['system_number'].unique()
        # Get list of metrics
        metrics = ['P@10', 'R@50', 'r-precision', 'AP', 'nDCG@10', 'nDCG@20']
        
        # Initialize a dictionary to store p-values
        p_values = {}
        
        # Iterate over all pairs of systems
        for i in range(len(systems)):
            for j in range(i+1, len(systems)):
                system_i = systems[i]
                system_j = systems[j]
                p_values_pair = {}
                # Perform t-test for each metric
                for metric in metrics:
                    scores_i = results_df[results_df['system_number'] == system_i][metric].astype(float)
                    scores_j = results_df[results_df['system_number'] == system_j][metric].astype(float)
                    # Perform paired t-test
                    t_stat, p_value = ttest_rel(scores_i, scores_j)
                    p_values_pair[metric] = p_value
                p_values[(system_i, system_j)] = p_values_pair
        
        # Store the p-values
        self.p_values = p_values
        return p_values


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
        # Initialize stop words list and stemmer
        self.stop_words = set(stopwords.words('english'))
        self.ps = PorterStemmer()

        # Read and preprocess data
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
        # Convert to lowercase
        text = text.lower()
        # Remove punctuation and special characters
        text = re.sub(r'[^a-z\s]', '', text)
        # Tokenize
        tokens = nltk.word_tokenize(text)
        # Remove stop words
        tokens = [word for word in tokens if word not in self.stop_words]
        # Stemming
        tokens = [self.ps.stem(word) for word in tokens]
        return tokens

    def calculate_mi_chi2(self):
        # Initialize data structures
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

        # Get all labels
        labels_set = set(corpus_docs.keys())

        for label in labels_set:
            # Number of documents in the current corpus
            N_c = corpus_docs[label]
            # Number of documents in other corpora
            N_not_c = total_docs - N_c

            # Document frequency of each word in the current corpus
            token_doc_freq_c = Counter()
            for tokens in corpus_tokens[label]:
                token_doc_freq_c.update(tokens)

            # Document frequency of each word in other corpora
            token_doc_freq_not_c = Counter()
            for other_label in labels_set - {label}:
                for tokens in corpus_tokens[other_label]:
                    token_doc_freq_not_c.update(tokens)

            # Set of all tokens
            tokens = all_tokens

            for token in tokens:
                # Calculate N11
                N11 = token_doc_freq_c.get(token, 0)
                # Calculate N01
                N01 = token_doc_freq_not_c.get(token, 0)
                # Calculate N10
                N10 = N_c - N11
                # Calculate N00
                N00 = N_not_c - N01
                # Total number of documents
                N = total_docs

                # To avoid mathematical errors due to zero values, add a small epsilon
                epsilon = 1e-10

                # Calculate Mutual Information (MI)
                MI = 0
                # Calculate each term; skip if frequency is zero
                # First term
                if N11 > 0:
                    MI += (N11 / N) * (np.log2((N * N11) / ((N11 + N10) * (N11 + N01) + epsilon)))
                # Second term
                if N01 > 0:
                    MI += (N01 / N) * (np.log2((N * N01) / ((N01 + N00) * (N11 + N01) + epsilon)))
                # Third term
                if N10 > 0:
                    MI += (N10 / N) * (np.log2((N * N10) / ((N11 + N10) * (N10 + N00) + epsilon)))
                # Fourth term
                if N00 > 0:
                    MI += (N00 / N) * (np.log2((N * N00) / ((N01 + N00) * (N10 + N00) + epsilon)))

                self.mi_scores[label][token] = MI

                # Calculate χ² (Chi-squared)
                numerator = (N11 * N00 - N10 * N01) ** 2 * N
                denominator = (N11 + N01) * (N11 + N10) * (N10 + N00) * (N01 + N00) + epsilon
                chi2 = numerator / denominator
                self.chi2_scores[label][token] = chi2

    def save_mi_chi2_results(self):
        # Sort words by score for each corpus
        for label in set(self.labels):
            # MI sorting
            mi_sorted = sorted(self.mi_scores[label].items(), key=lambda x: x[1], reverse=True)
            # Save MI results
            with open(f'{label}_mi.txt', 'w', encoding='utf-8') as f:
                for token, score in mi_sorted:
                    f.write(f'{token},{score}\n')

            # χ² sorting
            chi2_sorted = sorted(self.chi2_scores[label].items(), key=lambda x: x[1], reverse=True)
            # Save χ² results
            with open(f'{label}_chi2.txt', 'w', encoding='utf-8') as f:
                for token, score in chi2_sorted:
                    f.write(f'{token},{score}\n')

    def run_lda(self):
        # Create dictionary
        self.dictionary = corpora.Dictionary(self.texts)
        # Create corpus
        self.corpus = [self.dictionary.doc2bow(text) for text in self.texts]
        # Train LDA model
        self.lda_model = models.LdaModel(corpus=self.corpus, id2word=self.dictionary, num_topics=self.num_topics, passes=10, random_state=42)
        # Get topic distribution for each document
        doc_topics = self.lda_model.get_document_topics(self.corpus)
        # Initialize dictionary to store topic distributions for each corpus
        corpus_topic_distributions = defaultdict(list)

        for idx, doc_topic in enumerate(doc_topics):
            label = self.labels[idx]
            # Convert topic distribution to array
            topic_prob = np.zeros(self.num_topics)
            for topic_id, prob in doc_topic:
                topic_prob[topic_id] = prob
            corpus_topic_distributions[label].append(topic_prob)

        # Calculate average topic probability for each corpus
        for label, topic_distributions in corpus_topic_distributions.items():
            # Convert list to matrix
            topic_matrix = np.array(topic_distributions)
            # Calculate mean probability for each topic
            topic_avg = np.mean(topic_matrix, axis=0)
            self.corpus_topic_averages[label] = topic_avg

    def display_lda_results(self):
        for label, topic_avg in self.corpus_topic_averages.items():
            # Find the topic number with the highest average score
            top_topic = np.argmax(topic_avg)
            print(f"The most relevant topic number for {label} corpus: {top_topic}")

            # Get the top 10 words and their probabilities for that topic
            top_words = self.lda_model.show_topic(top_topic, topn=10)
            print(f"Top 10 words and their probabilities for the most relevant topic in {label} corpus:")
            for word, prob in top_words:
                print(f"{word}: {prob}")
            print("\n")

        # Convert average topic probabilities to DataFrame
        topic_avg_df = pd.DataFrame(self.corpus_topic_averages, index=[f"Topic_{i}" for i in range(self.num_topics)])
        # Print the average topic scores table
        print("Average topic scores for each corpus:")
        print(topic_avg_df)


class TextClassifier:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        self.vectorizer = None
        self.model = None
        self.category_mapping = {}
        self.inverse_category_mapping = {}
    
    def load_data(self, data_file):
        # Read the entire dataset
        data = pd.read_csv(data_file, sep='\t', header=0, names=['id', 'category', 'text'])

        # Map categories to numerical IDs
        categories = data['category'].unique()
        self.category_mapping = {category: idx for idx, category in enumerate(categories)}
        self.inverse_category_mapping = {idx: category for category, idx in self.category_mapping.items()}
        
        # Convert categories to numerical IDs
        data['category_id'] = data['category'].map(self.category_mapping)

        # Shuffle and split the dataset
        self.train_data, self.dev_data = train_test_split(
            data, test_size=0.1, random_state=42, stratify=data['category_id'])

    def preprocess_text(self, text):
        # Convert to lowercase
        text = text.lower()
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        # Remove @mentions and keep hashtags (optional)
        text = re.sub(r'@\w+', '', text)
        # Keep words from hashtags (e.g., convert #car to car)
        text = re.sub(r'#', '', text)
        # Remove punctuation and numbers
        text = re.sub(r'[^a-z\s]', '', text)
        # Tokenize
        tokens = nltk.word_tokenize(text)
        # Remove stop words (optional: keep stop words to improve the model)
        # tokens = [word for word in tokens if word not in self.stop_words]
        # Apply lemmatization or stemming (choose one)
        tokens = [self.lemmatizer.lemmatize(word) for word in tokens]
        # Reconstruct the string
        return ' '.join(tokens)
    
    def preprocess_data(self):
        # Preprocess training and development data
        self.train_data['clean_text'] = self.train_data['text'].apply(self.preprocess_text)
        self.dev_data['clean_text'] = self.dev_data['text'].apply(self.preprocess_text)

    def extract_features(self, method='bow', ngram_range=(1,1), use_chi2=False, k=1000):
        # Extract features, default is BOW
        if method == 'bow':
            self.vectorizer = CountVectorizer(ngram_range=ngram_range)
        elif method == 'tfidf':
            self.vectorizer = TfidfVectorizer(ngram_range=ngram_range)
        else:
            raise ValueError("Unsupported feature extraction method.")
        
        # Fit the vectorizer only on the training data
        self.X_train = self.vectorizer.fit_transform(self.train_data['clean_text'])
        # Transform the development data
        self.X_dev = self.vectorizer.transform(self.dev_data['clean_text'])
        
        # Feature selection (optional)
        if use_chi2:
            self.selector = SelectKBest(chi2, k=k)
            self.X_train = self.selector.fit_transform(self.X_train, self.train_data['category_id'])
            self.X_dev = self.selector.transform(self.X_dev)
        else:
            self.selector = None  # Set to None if not using feature selector
        
        # Get labels
        self.y_train = self.train_data['category_id']
        self.y_dev = self.dev_data['category_id']


    def train_model(self, model_type='svm', C=1000):
        # Train model, default is SVM
        if model_type == 'svm':
            self.model = LinearSVC(C=C, max_iter=1000, dual=False, random_state=42)
        else:
            raise ValueError("Unsupported model type.")
        
        self.model.fit(self.X_train, self.y_train)
    
    def evaluate_model(self, split='dev'):
        # Choose evaluation set
        if split == 'train':
            X = self.X_train
            y_true = self.y_train
        elif split == 'dev':
            X = self.X_dev
            y_true = self.y_dev
        else:
            raise ValueError("Unsupported split.")

        # Predict on the specified set
        y_pred = self.model.predict(X)
        
        # Compute evaluation metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=list(self.category_mapping.values()), average=None)
        macro_precision = precision_score(y_true, y_pred, average='macro')
        macro_recall = recall_score(y_true, y_pred, average='macro')
        macro_f1 = f1_score(y_true, y_pred, average='macro')
        
        # Print results
        print(f"Evaluation set: {split}")
        for idx, category in self.inverse_category_mapping.items():
            print(f"Class {category}: Precision={precision[idx]:.4f}, Recall={recall[idx]:.4f}, F1-score={f1[idx]:.4f}")
        print(f"Macro-average: Precision={macro_precision:.4f}, Recall={macro_recall:.4f}, F1-score={macro_f1:.4f}")
        
        # Plot confusion matrix (optional)
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', xticklabels=self.category_mapping.keys(),
                    yticklabels=self.category_mapping.keys(), cmap='Blues')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.title(f'Confusion Matrix ({split} set)')
        plt.show()
        
        return precision, recall, f1, macro_precision, macro_recall, macro_f1, y_pred

    def identify_misclassifications(self):
        # Predict on the development set
        y_pred = self.model.predict(self.X_dev)
        misclassified_indices = np.where(self.y_dev != y_pred)[0]
        
        # Get the first three misclassified examples
        print("Misclassified examples:")
        for idx in misclassified_indices[:3]:
            print("Text:", self.dev_data.iloc[idx]['text'])
            actual_label = self.inverse_category_mapping[self.y_dev.iloc[idx]]
            predicted_label = self.inverse_category_mapping[y_pred[idx]]
            print("Actual Label:", actual_label)
            print("Predicted Label:", predicted_label)
            print("---")
    
    def save_results(self, system_name, split, y_true, y_pred, filename='classification.csv'):
        # Define column names
        columns = ['system', 'split', 'p-pos', 'r-pos', 'f-pos',
                   'p-neg', 'r-neg', 'f-neg', 'p-neu', 'r-neu', 'f-neu',
                   'p-macro', 'r-macro', 'f-macro']

        # Check if the file exists; if not, write the header
        import os
        if not os.path.exists(filename):
            with open(filename, mode='w', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerow(columns)

        # Compute evaluation metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=list(self.category_mapping.values()), zero_division=0)
        macro_precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
        macro_recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
        macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

        # Get class indices
        pos_idx = self.category_mapping['positive']
        neg_idx = self.category_mapping['negative']
        neu_idx = self.category_mapping['neutral']

        # Prepare data
        data = [
            system_name,
            split,
            precision[pos_idx], recall[pos_idx], f1[pos_idx],
            precision[neg_idx], recall[neg_idx], f1[neg_idx],
            precision[neu_idx], recall[neu_idx], f1[neu_idx],
            macro_precision, macro_recall, macro_f1
        ]

        # Write to CSV file
        with open(filename, mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(data)



    def evaluate_and_save_results(self, system_name):
        for split in ['train', 'dev']:
            precision, recall, f1, macro_precision, macro_recall, macro_f1, y_pred = self.evaluate_model(split=split)
            y_true = self.y_train if split == 'train' else self.y_dev
            self.save_results(system_name, split, y_true, y_pred)

    
    def run(self):
        # Preprocess data
        self.preprocess_data()

        # ========== Baseline Model ==========
        print("==== Baseline Model ====")
        self.extract_features(method='bow', ngram_range=(1,1), use_chi2=False)
        # Save vectorizer and selector for the baseline model
        self.baseline_vectorizer = self.vectorizer
        self.baseline_selector = self.selector  # Should be None for the baseline model
        self.train_model(model_type='svm', C=1000)
        self.baseline_model = self.model
        # Evaluate and save results
        self.evaluate_and_save_results('baseline')

        # ========== Improved Model ==========
        print("==== Improved Model ====")
        self.extract_features(method='bow', ngram_range=(1,2), use_chi2=True, k=2000)
        # Save vectorizer and selector for the improved model
        self.improved_vectorizer = self.vectorizer
        self.improved_selector = self.selector
        self.train_model(model_type='svm', C=500)
        self.improved_model = self.model
        # Evaluate and save results
        self.evaluate_and_save_results('improved')

        # Evaluate on the test set
        self.evaluate_on_test_set()



    def evaluate_on_test_set(self):
        # Load test dataset
        test_data = pd.read_csv('ttds_2024_cw2_test.txt', sep='\t', header=0, names=['id', 'category', 'text'])
        test_data['category_id'] = test_data['category'].map(self.category_mapping)
        test_data['clean_text'] = test_data['text'].apply(self.preprocess_text)
        y_test = test_data['category_id']
        
        # Evaluate for both baseline and improved models
        for system_name, model, vectorizer, selector in [
            ('baseline', self.baseline_model, self.baseline_vectorizer, self.baseline_selector),
            ('improved', self.improved_model, self.improved_vectorizer, self.improved_selector)]:

            # Transform test data using the corresponding vectorizer
            X_test = vectorizer.transform(test_data['clean_text'])

            # Apply feature selector if it exists
            if selector is not None:
                X_test = selector.transform(X_test)

            # Make predictions
            y_pred_test = model.predict(X_test)

            # Compute evaluation metrics
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_test, y_pred_test, labels=list(self.category_mapping.values()), zero_division=0)
            macro_precision = precision_score(y_test, y_pred_test, average='macro', zero_division=0)
            macro_recall = recall_score(y_test, y_pred_test, average='macro', zero_division=0)
            macro_f1 = f1_score(y_test, y_pred_test, average='macro', zero_division=0)

            # Save test set results
            self.save_results(system_name, 'test', y_test, y_pred_test)

            # Print test set results
            print(f"==== Test Set Evaluation Results ({system_name}) ====")
            for idx, category in self.inverse_category_mapping.items():
                print(f"Class {category}: Precision={precision[idx]:.4f}, Recall={recall[idx]:.4f}, F1-score={f1[idx]:.4f}")
            print(f"Macro-average: Precision={macro_precision:.4f}, Recall={macro_recall:.4f}, F1-score={macro_f1:.4f}")



if __name__ == '__main__':
    # # Information Retrieval Evaluation Part
    # ir_eval = InformationRetrievalEvaluation(system_results_file='ttdssystemresults.csv', qrels_file='qrels.csv')
    # ir_eval.load_data()
    # ir_eval.evaluate()
    # ir_eval.save_results(output_file='ir_eval.csv')
    # # Perform t-tests
    # p_values = ir_eval.perform_t_tests()
    # # Optionally, print the p-values
    # for systems_pair, metrics_pvalues in p_values.items():
    #     system_i, system_j = systems_pair
    #     print(f"\nT-test between System {system_i} and System {system_j}:")
    #     for metric, p_value in metrics_pvalues.items():
    #         significance = 'significant' if p_value < 0.05 else 'not significant'
    #         print(f"  {metric}: p-value = {p_value:.4f} ({significance})")

    # Text Analysis Part
    # text_analysis = TextAnalysis(corpus_file='bible_and_quran.tsv')
    # text_analysis.load_and_preprocess_data()
    # text_analysis.calculate_mi_chi2()
    # text_analysis.save_mi_chi2_results()
    # text_analysis.run_lda()
    # text_analysis.display_lda_results()

    # Instantiate the classifier
    classifier = TextClassifier()
    
    # Load the dataset (ensure the path and filename are correct)
    classifier.load_data('train.txt')
    
    # Run the classification process
    classifier.run()
