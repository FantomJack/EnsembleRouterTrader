from __future__ import annotations
from src.data.market_frame import MarketFrame
from feature_generator import FeatureGeneratorInterface

class FeaturePipeline:
    def __init__(self, *generators: FeatureGeneratorInterface):
        self.generators = generators

    def run(self, frame: MarketFrame,
    ) -> MarketFrame:
        for generator in self.generators:
            generator.transform(frame)
        return frame

pipeline = FeaturePipeline(

    # EMAFeatures(),

    # RSIFeatures(),

    # ATRFeatures(),

    # VolumeFeatures(),

    # TrendFeatures(),
    # MomentumFeatures(),
    # VolatilityFeatures(),

)

