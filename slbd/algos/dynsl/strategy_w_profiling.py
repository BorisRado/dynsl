from slbd.algos.base_strategy import BaseStrategy


class DynSLStrategyWProfiling(BaseStrategy):

    def __init__(self, end_epoch_weight_integration_strategy: str, **kwargs):
        super().__init__(**kwargs)
