import pandas as pd
import pickle


with open("./dataset/processed/rect.pkl", "rb") as f:
    df = pickle.load(f)
    print(df)