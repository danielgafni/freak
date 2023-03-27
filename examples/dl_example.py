from time import sleep
from typing import List

from pydantic import BaseModel

from freak import control
from logging import basicConfig


class Head(BaseModel):
    activation: str = "relu"


class Model(BaseModel):
    hidden_dim: List[int] = [128, 256, 512]
    head: Head = Head()


class Checkpoints(BaseModel):
    every_epochs: int = 2
    save_dir: str = "checkpoints"


class State(BaseModel):
    lr: float = 1e-3
    checkpoints: Checkpoints = Checkpoints()
    model: Model = Model()


def epoch_loop(config: State, current_epoch: int):
    # training our great model here
    sleep(5)

    if current_epoch % config.checkpoints.every_epochs == 0:
        print(f"Saving checkpoint after epoch {current_epoch} to {config.checkpoints.save_dir}")


if __name__ == "__main__":
    basicConfig(level="INFO")

    state = State()
    control(state)

    current_epoch = 0

    while True:
        print(f"state: {state}")
        epoch_loop(state, current_epoch)
        current_epoch += 1
