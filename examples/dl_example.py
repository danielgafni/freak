from logging import basicConfig
from time import sleep
from typing import List

from pydantic import BaseModel

from freak import control


class Head(BaseModel):
    activation: str = "relu"


class Model(BaseModel):
    hidden_dim: List[int] = [128, 256, 512]
    head: Head = Head()


class Checkpoints(BaseModel):
    every_epochs: int = 2
    save_dir: str = "checkpoints"


class State(BaseModel):
    training_stopped: bool = False
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
    server = control(state)

    current_epoch = 0

    while not state.training_stopped:
        print(f"state: {state}")
        epoch_loop(state, current_epoch)
        current_epoch += 1
    else:
        print("Training stopped!")
        server.stop()
