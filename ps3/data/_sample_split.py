from pandas.util import hash_pandas_object
import numpy as np

# TODO: Write a function which creates a sample split based in some id_column and training_frac.
# Optional: If the dtype of id_column is a string, we can use hashlib to get an integer representation.
def create_sample_split(df, id_column, training_frac=0.8):
    """Create sample split based on ID column.

    Parameters
    ----------
    df : pd.DataFrame
        Training data
    id_column : str
        Name of ID column
    training_frac : float, optional
        Fraction to use for training, by default 0.9

    Returns
    -------
    pd.DataFrame
        Training data with sample column containing train/test split based on IDs.
    """

    hashed = hash_pandas_object(df[id_column].astype(str), index=False)

    df["sample"] = np.where(
        (hashed % 100) >= (training_frac * 100),
        "test",
        "train"
    )

    return df

if __name__ == "__main__":
    from _load_transform import load_transform

    df = load_transform()
    df = create_sample_split(df, "IDpol")

    train_count = np.sum(df["sample"] == "train")
    test_count = np.sum(df["sample"] == "test")
    print(f"Train: {train_count}")
    print(f"Test: {test_count}")
    print(f"Proportion train: {train_count / (train_count + test_count)}")