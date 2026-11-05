"""Clean and sample transaction/complaint narratives before embedding."""
import pandas as pd


def preprocess(frame: pd.DataFrame, text_column="consumer_complaint_narrative") -> pd.DataFrame:
    data=frame.copy(); data[text_column]=data[text_column].fillna("").astype(str).str.replace(r"\s+"," ",regex=True).str.strip()
    return data[data[text_column].str.len() >= 40].drop_duplicates(subset=[text_column])

