class DataProviderError(Exception):
    """Base class for market-data problems."""


class SymbolNotFound(DataProviderError):
    pass


class ProviderUnavailable(DataProviderError):
    """Network failure, rate limit, or provider outage."""


class InsufficientData(Exception):
    """Not enough clean history to train/evaluate a model."""
