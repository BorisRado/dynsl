from slbd.algos.base_numpy_client import BaseNumpyClientAdapter


class SplitFedClientAdapter(BaseNumpyClientAdapter):
    def get_properties(self, config):
        _ = (config, )
        return {}
