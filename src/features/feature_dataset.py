import pandas as pd


@dataclass(slots=True)
class FeatureDataset:

    date: pd.DataFrame

    feature_columns: list[str]

    target_column: str

    metadata: dict

