from typing import Callable, Union
import pandas as pd
import argparse
import traceback
import regex as re
from topicgpt_python.utils import *
import regex
import json
from gensim.models.coherencemodel import CoherenceModel
from anytree import Node, RenderTree
from sklearn.feature_extraction.text import CountVectorizer
from gensim import corpora
from .ctfidf import ClassTfidfTransformer
from .metrics import *
from .utils import *
from .correction import topic_parser
from sklearn.utils import Bunch
import requests


def remove_emojis(text):
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U0001F1F2-\U0001F1F4"  # Macau flag
        u"\U0001F1E6-\U0001F1FF"  # flags
        u"\U0001F600-\U0001F64F"
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U0001f926-\U0001f937"
        u"\U0001F1F2"
        u"\U0001F1F4"
        u"\U0001F620"
        u"\u200d"
        u"\u2640-\u2642"
        "]+", flags=re.UNICODE)

    text = emoji_pattern.sub(r' ', text)
    return text

def remove_html_code(text):
    text = re.sub(r'<[^>]*>', '', text)
    return text

def preprocessAdvanced(text, keep_emojis=False, keep_hashtags=False, keep_mentions=False, cased=True):
    if not keep_emojis:
        text = remove_emojis(text)
    if not keep_hashtags:
        text = re.sub(r'#\w*', ' ', text)
    if not keep_mentions:
        text = re.sub(r'@\w*', ' ', text)

    # Remove URLS
    text = re.sub(r'https?://[^\s<>"]+|www\.[^\s<>"]+', ' ', text)

    # Remove Special Characters
    text = re.sub(r'["\/\\\:\-\[\]]', ' ', text)

    # Remove Extra Spaces, New Lines, Tabs
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\t', ' ', text)
    text = re.sub(r'\s+', ' ', text)

    # Remove leading and trailing whitespaces
    text = text.strip()

    if not cased:
        text = text.lower()

    return text

def create_topic_representations_c_tf_idf(data,topic_file,out_file, out_file_grouped, out_file_metrics,out_file_topics_id, stop_words: Union['English', 'German'], min_df: int = 5, max_df: float = 0.8, allow_multi_topics: bool = False, preprocess_fn: Callable[[str], str] = None):
    """
    Create topic representations using C-TF-IDF

    Parameters:
    - data: DataFrame
    - out_file: str
    - stop_words: 'English' or 'German'
    - min_df: int, minimum document frequency
    - max_df: float, maximum document frequency
    - allow_multi_topics: bool, whether to allow multiple topics per document
    - preprocess_fn: Callable, function to preprocess text data
    """

    preprocess_fn = preprocess_fn or ( lambda x: preprocessAdvanced(remove_html_code(x)) )

    topics_root = TopicTree().from_topic_list(topic_file, from_file=True)

    ordered_topics = topics_root.get_root_descendants_name()
    ordered_topics = list(dict.fromkeys(ordered_topics))  # remove duplicates while preserving order
    df = pd.read_json(data, lines=True)

    error, hallucinated = topic_parser(topics_root, df, False)
    reprompt_idx = error + hallucinated

    if len(reprompt_idx) > 0:
        print("Number of rows with Error or Hallucinated Topics:", len(reprompt_idx))
        print("Considered as Outliers, creating topic for outliers")
        for i in reprompt_idx:
            df.at[i, "responses"] = "[0] Outliers: Error or Hallucinated."

    def parse_topic_name(response):
        topics = []
        main_pattern = regex.compile(r"^\[(\d+)\] ([\w\s]+):(.+)")
        for line in response.strip().split("\n"):
            line = line.strip()
            match = regex.match(main_pattern, line)
            if match:
                if ":" in line:  # add the topic to the tre
                    lvl, name = (
                        int(match.group(1)),
                        match.group(2).strip()
                    )
                    clean_name = name
                    topics.append(clean_name)

        if not topics:
            topics = ["Outliers"]
        return topics

    df["topics"] = df["responses"].apply(parse_topic_name)

    # first topic
    if not allow_multi_topics:
        # pick first topic
        df["topics"] = df["topics"].apply(lambda x: x[0])

    else:
        # create seperate records / rows for each topic of a document
        df = df.explode("topics")
        
    # Topics to Identifier
    topics_to_id = {}
    topic_counts = {}
    for i, topic in enumerate(ordered_topics):
        topics_to_id[topic] = i
        topic_counts[topic] = 0
    # Add outliers
    topics_to_id["Outliers"] = len(ordered_topics)
    topic_counts["Outliers"] = 0
    print(topics_to_id)

    df["target"] = df["topics"].apply(lambda topic: topics_to_id[topic])

    # toic counts
    for i, row in df.iterrows():
        topic = row["topics"]
        topic_counts[topic] = topic_counts[topic] + 1

    print("Topic Counts:", topic_counts)
    df["text"] = df["text"].apply(lambda x: preprocess_fn(x))

    dataset = Bunch(data=df["text"].to_list(), target=df["target"].to_list(), target_names=list(topics_to_id.keys()))

    docs = pd.DataFrame({"Document": dataset.data, "Class": dataset.target})

    print("Number of Documents:", len(docs))
    docs_per_class = docs.groupby(['Class'], as_index=False).agg({'Document': ' '.join})

    # FIX Empty Topics: Ensure all Classes / Topics are represented
    all_classes = pd.DataFrame({'Class':  range(len(dataset.target_names))})
    docs_per_class = all_classes.merge(docs_per_class, on='Class', how='left')
    docs_per_class['Document'] = docs_per_class['Document'].fillna('')

    print(docs_per_class)
    if stop_words == 'German':
        stop_words = requests.get("https://raw.githubusercontent.com/stopwords-iso/stopwords-de/master/stopwords-de.txt").text.split("\n")
    elif stop_words == 'English':
        stop_words = 'english'
    else:
        raise ValueError("Stop words must be either 'English' or 'German'")
    
    count_vectorizer = CountVectorizer(stop_words=stop_words, min_df=min_df, max_df=max_df).fit(docs_per_class['Document'])
    count = count_vectorizer.transform(docs_per_class['Document'])
    words = count_vectorizer.get_feature_names_out()
    analyzer = count_vectorizer.build_analyzer()

    c_tf_idf_vectorizer = ClassTfidfTransformer()
    c_tf_idf_vectorizer.fit(count)
    c_tf_idf = c_tf_idf_vectorizer.transform(count).toarray()

    words_per_class = {dataset.target_names[label]: [words[index] for index in c_tf_idf[label].argsort()[-10:]] for label in docs_per_class.Class}

    # persist to json
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(words_per_class, f)

    docs_per_class.to_json(out_file_grouped, orient="records", lines=True)

    with open(out_file_topics_id, "w", encoding="utf-8") as f:
        json.dump(topics_to_id, f)

    tokens = [analyzer(doc) for doc in dataset.data]
    dictionary = corpora.Dictionary(tokens)
    corpus = [dictionary.doc2bow(token) for token in tokens]

    # create list of lists of topic words
    topic_words = []
    for topic in dataset.target_names:
        topic_words.append(words_per_class[topic])

    # Calculated Metrics
    metrics = {}
    metrics_cv= {}
    cm = CoherenceModel(topics=topic_words, texts=tokens, dictionary=dictionary, corpus=corpus, coherence='c_v')
    metrics_cv["Model_cv"] = cm.get_coherence()
    metrics_cv["Topics_cv"] = cm.get_coherence_per_topic()
    td = TopicDiversity()
    rbo = InvertedRBO()
    metrics_td = {}
    metrics_td["Topic_Diversity"] = td.score({"topics": topic_words})
    metrics_td["inverted_rbo"] = rbo.score({"topics": topic_words})

    metrics["CoherenceModel"] = metrics_cv
    metrics["TopicDiversity"] = metrics_td
    
    # Save Metrics to file
    with open(out_file_metrics, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)

    return {"topics": topic_words, "texts": tokens, "dictionary": dictionary, "corpus": corpus, "topics_to_id": topics_to_id}

if __name__ == "__main__":
    from topicgpt_python import *
    import yaml

    with open("config.yml", "r") as f:
        config = yaml.safe_load(f)

    create_topic_representations_c_tf_idf(
        config["correction"]["output"],"representations.json",config["generation"]["topic_output"])