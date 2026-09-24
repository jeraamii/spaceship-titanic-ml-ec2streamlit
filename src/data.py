"""Load and split the Spaceship Titanic dataset."""

import pandas as pd
from sklearn.model_selection import train_test_split

# Default local path (used when running pipeline.py on SageMaker Notebook)
DATA_PATH = "train.csv"

CLASS_NAMES = ["Not Transported", "Transported"]

PROCESSED_FEATURES = [
    "HomePlanet", "CryoSleep", "Destination",
    "Age", "VIP",
    "RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck",
    "CabinDeck", "CabinSide", "TotalSpend",
]


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Feature engineering + missing value handling."""
    df = df.copy()

    # Split Cabin into Deck / Side
    cabin_split = df["Cabin"].str.split("/", expand=True)
    df["CabinDeck"] = cabin_split[0]
    df["CabinSide"] = cabin_split[2]
    df.drop(columns=["Cabin"], inplace=True)

    # Boolean -> int
    for col in ["CryoSleep", "VIP"]:
        df[col] = df[col].fillna(False).astype(int)

    # Numeric -> fill with median
    numeric_cols = ["Age", "RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())

    # Categorical -> fill with mode then encode
    cat_cols = ["HomePlanet", "Destination", "CabinDeck", "CabinSide"]
    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode()[0])
        df[col] = df[col].astype("category").cat.codes

    # Total spending feature
    spend_cols = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
    df["TotalSpend"] = df[spend_cols].sum(axis=1)

    return df


def load_dataset(path: str = DATA_PATH) -> pd.DataFrame:
    """Load the Spaceship Titanic CSV."""
    return pd.read_csv(path)


def split_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """Preprocess and split. Returns (X_train, X_test, y_train, y_test)."""
    df = preprocess(df)
    X = df[PROCESSED_FEATURES]
    y = df["Transported"].astype(int)
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
