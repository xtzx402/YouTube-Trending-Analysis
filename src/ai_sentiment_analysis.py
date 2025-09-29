import os
import logging
import pandas as pd
from transformers import pipeline
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError
from src.config import DB_URI

#import sys, os
#sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
#from config import DB_URI

def init_emotion_model():
    """Load Hugging Face emotion classification pipeline."""
    logging.info("Loading Hugging Face emotion model...")
    return pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base",
        top_k=1,
        truncation=True
    )


def analyze_emotion(text, classifier):
    """Run emotion classification on text.

    Args:
        text (str): Input text (title + tags + description).
        classifier: Hugging Face pipeline.

    Returns:
        str: Predicted emotion label or 'neutral' if empty.
    """
    if not text or str(text).strip() == "":
        return "neutral"
    try:
        result = classifier(text)[0][0]
        return result["label"]
    except Exception as e:
        logging.error(f"Emotion analysis failed for text: {e}")
        return "error"


def run_emotion_analysis(batch_size: int = 200, output_dir: str = "outputs"):
    """Perform sentiment/emotion analysis on cleaned YouTube videos.

    Reads `us_videos` table, applies Hugging Face model, and writes results
    into `video_emotion` table. If insertion fails, saves to CSV fallback.

    Args:
        batch_size (int): Number of rows per processing batch.
        output_dir (str): Directory for fallback CSV outputs.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Connect to database
    try:
        engine = create_engine(DB_URI, echo=False)
        usvideos = pd.read_sql(
            "SELECT video_id, title, tags_clean, description FROM us_videos", 
            con=engine
        )
        logging.info(f"Loaded {len(usvideos)} rows from us_videos table.")
    except SQLAlchemyError as e:
        logging.error(f"Failed to load us_videos table: {e}")
        return

    classifier = init_emotion_model()
    total_rows = len(usvideos)
    start = 0

    # Batch processing
    while start < total_rows:
        end = min(start + batch_size, total_rows)
        chunk = usvideos.iloc[start:end].copy()

        # Prepare NLP input (title + tags + first 500 chars of description)
        chunk["nlp_text"] = (
            chunk["title"].fillna("") + " " +
            chunk["tags_clean"].fillna("") + " " +
            chunk["description"].fillna("").str[:500]
        )

        # Apply emotion model
        chunk["emotion"] = chunk["nlp_text"].apply(lambda x: analyze_emotion(x, classifier))

        # Keep only video_id and emotion
        df_emotion = chunk[["video_id", "emotion"]]

        # Write to database
        try:
            df_emotion.to_sql(
                "video_emotion",
                con=engine,
                if_exists="append" if start > 0 else "replace",
                index=False
            )
            logging.info(f"Processed rows {start} - {end}")
        except SQLAlchemyError as e:
            logging.error(f"Failed to insert rows {start}-{end}: {e}")
            fallback_path = os.path.join(output_dir, f"failed_emotion_{start}_{end}.csv")
            df_emotion.to_csv(fallback_path, index=False)
            logging.info(f"Saved failed batch to {fallback_path}")

        start = end

    logging.info("All chunks processed and saved into video_emotion")


def run_sentiment():
    """Wrapper for main.py integration."""
    run_emotion_analysis()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    run_emotion_analysis()
