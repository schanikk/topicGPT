import os
from pydantic import Field
from topicgpt_python import (
    sample_data,
    generate_topic_lvl1,
    generate_topic_lvl2,
    refine_topics,
    assign_topics,
    correct_topics, 
    create_topic_representations_c_tf_idf
    )
import yaml
from pydantic_settings import BaseSettings, SettingsConfigDict
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer
import pymysql
import time


from preprocessing.preprocessor import BERTBasedStrategy

class DBConfig(BaseSettings):
    host: str = Field(alias="DB_HOST")
    port: int = Field(alias="DB_PORT")
    user: str = Field(alias="DB_USER")
    password: str = Field(alias="DB_PASSWORD")
    database: str = Field(alias="DB_DATABASE")

    model_config = SettingsConfigDict(
        env_file=".env.db",
        env_file_encoding="utf-8"
        )


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env.remote",
        env_file_encoding="utf-8",
    )

    # Define your settings here
    REMOTE_BASE_URL: str = Field(
        default="localhost:8000",
        description="Base URL for the TopicGPT API"
        
    )
    REMOTE_API_KEY: str = Field(
        default="",
        description="API key for the TopicGPT service"
    )
    OPENAI_API_KEY: str = Field(
        default="",
        description="API key for OpenAI service"
    )


def load_configs():

    #logger.info\(.*\)
    with open("config.yml", "r") as f:
        config = yaml.safe_load(f)

    settings = Settings()
    #logger.info\(.*\)

    db_settings = DBConfig()
    return config, settings, db_settings

def load_data_set(config, settings):
    """
    Load a dataset from a MySQL database using PyMySQL and save it as a JSON file.
    
    Args:
        config (BaseSettings): A Pydantic settings model with the database configuration.
    """
    data_path = config["data_sample"]
    query = """
    SELECT id, Date(date) as date, message as text, source_id FROM post WHERE date BETWEEN '2024-11-01' AND '2025-03-01';
    """

    #logger.info\(.*\)
    try:
        connection = pymysql.connect(
            host=settings.host,
            port=settings.port,
            user=settings.user,
            password=settings.password,
            database=settings.database,
        )
    except pymysql.MySQLError as e:
        #logger.error\(.*\)
        raise

    #logger.info\(.*\)
    try:
        df = pd.read_sql(query, connection)
    finally:
        connection.close()
        #logger.info\(.*\)

    if 'text' not in df.columns:
        #logger.error\(.*\)
        raise ValueError("Query result must contain a 'text' column.")
    
    bert_strategy = BERTBasedStrategy()
    df["text"] = df.text.apply(lambda x: bert_strategy.preprocess(x))
    df = df.dropna(subset=["text"])
    if df.empty:
        #logger.error\(.*\)
        raise ValueError("No valid documents found after preprocessing.")
    df = df.drop_duplicates().reset_index(drop=True)
    #logger.info\(.*\)
    df.to_json(data_path, orient="records", lines=True)
    #logger.info\(.*\)

    return df

def generate_diverse_subset(config: dict):
    """
    Generate a diverse subset of documents for topic modeling based on embeddings and clustering.
    
    Args:
        config (dict): Configuration dictionary with keys:
            - "data_sample": Path to the input CSV file containing documents in a 'text' column.
            - "subset_generation": Path to save the output subset CSV.
    """
    data_path = config.get("data_sample")
    output_path = config.get("generation_subset")

    if not data_path or not output_path:
        #logger.error\(.*\)
        raise ValueError("Both 'data_sample' and 'generation_subset' must be specified in config.")
    
    #logger.info\(.*\)
    df = pd.read_json(data_path, lines=True)
    if 'text' not in df.columns:
        #logger.error\(.*\)
        raise ValueError("Input file must contain a 'text' column.")
    
    texts = df['text'].tolist()
    #logger.info\(.*\)

    #logger.info\(.*\)
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

    #logger.info\(.*\)
    embeddings = model.encode(texts, show_progress_bar=True)

    n_clusters = min(300, len(texts))
    #logger.info\(.*\)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(embeddings)

    #logger.info\(.*\)
    df["cluster"] = labels
    def sample_group(group):
        n = min(10, len(group))
        n = max(n,5) if len(group) >= 5 else len(group)
        return group.sample(n=n, random_state=42)
    
    subset_df = df.groupby("cluster", group_keys=False).apply(sample_group).reset_index(drop=True)

    #logger.info\(.*\)
    subset_df.to_json(output_path, orient="records", lines=True)
    #logger.info\(.*\)

    return subset_df


def create_topic_generation_subset(config):
    #logger.info\(.*\)
    df = pd.read_json(config["data_sample"], lines=True)
    df = df.groupby("Date", as_index=False).sample(frac=0.05)
    df = df.reset_index(drop=True)
    df.to_json(config["generation_subset"], orient="records", lines=True)
    #logger.info\(.*\)
    return config

def run_experiment(config, settings):
    #logger.info\(.*\)

    os.environ["REMOTE_BASE_URL"] = settings.REMOTE_BASE_URL
    os.environ["REMOTE_API_KEY"] = settings.REMOTE_API_KEY
    os.environ["OPENAI_API_KEY"] = settings.OPENAI_API_KEY
    start_time = time.time()
    # generate_topic_lvl1(
    # config["connection"],
    # config["model"],
    # config["data_sample"],
    # config["generation"]["prompt"],
    # config["generation"]["seed"],
    # config["generation"]["output"],
    # config["generation"]["topic_output"],
    # verbose=config["verbose"],
    # use_basic_auth=True,
    # #batch_size=64,
    # #max_workers=64
    # )
    step_time = time.time()
    print(f"Topic generation took {step_time- start_time:.2f} seconds")
    #logger.info\(.*\)

    # if config["refining_topics"]:
    #     #logger.info\(.*\)
    #     refine_topics(
    #         config["connection"],
    #         config["model"],
    #         config["refinement"]["prompt"],
    #         config["generation"]["output"],
    #         config["generation"]["topic_output"],
    #         config["refinement"]["topic_output"],
    #         config["refinement"]["output"],
    #         verbose=config["verbose"],
    #         remove=config["refinement"]["remove"],
    #         mapping_file=config["refinement"]["mapping_file"],
    #         use_basic_auth=True,
    #     )
    # step_time = time.time()
    # print(f"Topic refinement took {step_time - start_time:.2f} seconds")
    #     #logger.info\(.*\)
    # # Optional: Generate subtopics
    # if config["generate_subtopics"]:
    #     #logger.info\(.*\)
    #     generate_topic_lvl2(
    #         config["connection"],
    #         config["model"],
    #         config["refinement"]["topic_output"],
    #         config["refinement"]["output"],
    #         config["generation_2"]["prompt"],
    #         config["generation_2"]["output"],
    #         config["generation_2"]["topic_output"],
    #         verbose=config["verbose"],
    #         use_basic_auth=True,
    #     )
        #logger.info\(.*\)
    step_time = time.time()
    print(f"Subtopic generation took {step_time - start_time:.2f} seconds")
    # Assignment
    #logger.info\(.*\)
    assign_topics(
        config["connection"],
        config["model"],
        config["data_sample"],
        config["assignment"]["prompt"],
        config["assignment"]["output"],
        config["generation_2"][
            "topic_output"
        ],  # TODO: change to generation_2 if you have subtopics, or config['refinement']['topic_output'] if you refined topics
        verbose=config["verbose"],
        api_key=None,
        use_basic_auth=True,
        max_workers=64
    )
    #logger.info\(.*\)
    step_time = time.time()
    print(f"Topic assignment took {step_time - start_time:.2f} seconds")
    # Correction
    #logger.info\(.*\)
    correct_topics(
        config["connection"],
        config["model"],
        config["assignment"]["output"],
        config["correction"]["prompt"],
        config["generation_2"][
            "topic_output"
        ],  # TODO: change to generation_2 if you have subtopics, or config['refinement']['topic_output'] if you refined topics
        config["correction"]["output"],
        verbose=config["verbose"],
        use_basic_auth=True,
        max_workers=64,
    )
    #logger.info\(.*\)
    step_time = time.time()
    print(f"Topic correction took {step_time - start_time:.2f} seconds")


if __name__ == "__main__":
    config, settings, db_settings = load_configs()
    config["verbose"] = True
    config["refining_topics"] = True

    # Load data from the database and save it as a JSON file
    #load_data_set(config, db_settings)
    # Create a subset of data for topic generation
    #generate_diverse_subset(config)

    # Run the experiment
    run_experiment(config, settings)