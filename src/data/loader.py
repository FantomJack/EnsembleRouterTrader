from pathlib import Path
import pandas as pd

from preprocessing.market_data import MarketData


class MarketDataLoader:

    REQUIRED_COLUMNS = ["Date", "Open", "High", "Low", "Close", "Volume"]

    def load_directory(
            self,
            directory: Path
    ) -> MarketData:
        frames = []

        for csv_file in sorted(directory.glob("*.csv")):

            df = pd.read_csv(csv_file)

            missing = set(self.REQUIRED_COLUMNS) - set(df.columns)

            if len(missing) > 0:
                raise ValueError(f"Missing columns: {missing}")

            df["Date"] = pd.to_datetime(df["Date"])

            df["ticker"] = csv_file.stem.upper()

            frames.append(df)

        data = (
            pd.concat(frames, ignore_index=True)
            .sort_values(["ticker", "Date"])
            .reset_index(drop=True)
        )

        return MarketData(data)