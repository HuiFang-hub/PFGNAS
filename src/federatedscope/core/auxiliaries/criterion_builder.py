import logging
import src.federatedscope.register as register

logger = logging.getLogger(__name__)

try:
    from torch import nn
    from src.federatedscope.nlp.loss import *
    from src.federatedscope.cl.loss import *
except ImportError:
    nn = None

try:
    from src.federatedscope.contrib.loss import *
except ImportError as error:
    logger.warning(
        f'{error} in `src.federatedscope.contrib.loss`, some modules are not '
        f'available.')


def get_criterion(criterion_type, device):
    """
    This function builds an instance of loss functions from: \
    "https://pytorch.org/docs/stable/nn.html#loss-functions",
    where the ``criterion_type`` is chosen from.

    Arguments:
        criterion_type: loss function type
        device: move to device (``cpu`` or ``gpu``)

    Returns:
        An instance of loss functions.
    """
    for func in register.criterion_dict.values():
        criterion = func(criterion_type, device)
        if criterion is not None:
            return criterion

    if isinstance(criterion_type, str):
        if hasattr(nn, criterion_type):
            return getattr(nn, criterion_type)()
        else:
            raise NotImplementedError(
                'Criterion {} not implement'.format(criterion_type))
    else:
        raise TypeError()
