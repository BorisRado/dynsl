from slbd.algos.dynsl.strategy import DynSLStrategy


class DynSLStrategyWProfiling(DynSLStrategy):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        assert self.server_optim_state_handling is None
