from abc import ABC, abstractmethod

from src.data.market_frame import MarketFrame

class FeatureGeneratorInterface(ABC):
    """
    Base class for every feature generator.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def transform(
        self,
        frame: MarketFrame,
    ) -> None:
        """
        Adds features directly into MarketFrame.
        """
        pass