from slbd.algos.base_numpy_client import BaseNumpyClientAdapter
from slbd.common.constants import MIN_NUMBER_OF_CLIENT_LAYERS


class SplitFedClientAdapter(BaseNumpyClientAdapter):
    def get_properties(self, config):
        config = super().get_properties(config)
        del config[MIN_NUMBER_OF_CLIENT_LAYERS]
        return config
