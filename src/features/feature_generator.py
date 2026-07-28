
class FeatureGeneratorInterface:
    """
    Base class for every feature generator.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @property
    @abstractmethod
    def produced_features(self) -> list[str]:
        """
        Names of columns that this generator produces.
        """
        ...

    @abstractmethod
    def transform(
        self,
        frame: MarketFrame,
    ) -> MarketFrame:
        """
        Adds new features into MarketFrame.
        """
        ...