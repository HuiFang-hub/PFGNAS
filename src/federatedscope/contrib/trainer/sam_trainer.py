from src.federatedscope.register import register_trainer
from src.federatedscope.core.trainers import BaseTrainer


def call_sam_trainer(trainer_type):
    if trainer_type == 'sam_trainer':
        from src.federatedscope.contrib.trainer.sam import SAMTrainer
        return SAMTrainer


register_trainer('sam_trainer', call_sam_trainer)
