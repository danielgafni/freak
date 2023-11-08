from typing import Optional

import lightning as L
from lightning.pytorch.utilities.distributed import rank_zero_only
from pydantic import BaseModel

from freak import Freak


class TrainingState(BaseModel):
    should_stop: bool = False


class TrainingStopCallback(L.Callback):
    """
    Callback which stops training when self.state.shoudl_stop is set to True.
    """

    def __init__(self, freak: Optional[Freak] = None):
        self.freak = freak if freak is not None else Freak(host="127.0.0.1")
        self.state = TrainingState()

    @rank_zero_only
    def on_train_start(self, trainer: "L.Trainer", pl_module: "L.LightningModule") -> None:
        self.freak.control(self.state)  # launch the Freak server in a background thread

    def on_train_epoch_end(self, trainer: "L.Trainer", pl_module: "L.LightningModule") -> None:
        self.state = trainer.strategy.broadcast(self.state, 0)

        if self.state.should_stop:
            # this triggers lightning to stop training
            trainer.should_stop = True
            trainer.strategy.barrier()

    @rank_zero_only
    def on_train_end(self, trainer: "L.Trainer", pl_module: "L.LightningModule") -> None:
        self.freak.stop()
