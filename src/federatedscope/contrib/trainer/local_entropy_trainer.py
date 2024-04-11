from src.federatedscope.register import register_trainer
from src.federatedscope.core.trainers import BaseTrainer


def call_local_entropy_trainer(trainer_type):
    if trainer_type == 'local_entropy_trainer':
        from src.federatedscope.contrib.trainer.local_entropy \
            import LocalEntropyTrainer
        return LocalEntropyTrainer


register_trainer('local_entropy_trainer', call_local_entropy_trainer)
