


class FeaturePipeline:
    def __init__(self, *generators: FeatureGeneratorInterface):
        self.generators = generators

    def transform(self, market_data: MarketData) -> pd.DataFrame:
        for generator in self.generators:
            market_data = generator.transform(market_data)
        return market_data









pipeline = FeaturePipeline(

    # EMAFeatures(),

    # RSIFeatures(),

    # ATRFeatures(),

    # VolumeFeatures(),

)

# TODO:
# market_data: pipeline.transform(market_data)