import json
from typing import List, Tuple

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
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
    lr: float = 1e-3
    checkpoints: Checkpoints = Checkpoints()
    model: Model = Model()


@pytest.fixture(scope="function")
def state() -> State:
    return State()


@pytest.fixture(scope="function")
def state_and_client(state: State) -> Tuple[State, TestClient]:
    app = FastAPI()
    client = TestClient(app)

    control(state, app, serve=False)

    return state, client


def test_full_state(state_and_client: Tuple[State, TestClient]):
    state, client = state_and_client
    init_state = state.copy(deep=True)
    assert json.loads(state.json()) == client.get("/get").json()

    # let's test these references will still work
    # after we update the upper objects
    def get_lr():
        return state.lr

    def get_activation():
        return state.model.head.activation

    assert get_lr() == 0.001
    assert get_activation() == "relu"

    new_state = State(
        lr=2.0,
        model=Model(hidden_dim=[256, 512, 1024], head=Head(activation="sigmoid")),
    )

    resp = client.patch("/set", json=json.loads(new_state.json()))
    assert resp.status_code == 200
    assert json.loads(new_state.json()) == json.loads(state.json())

    assert get_lr() == 2.0
    assert get_activation() == "sigmoid"

    resp = client.patch("/set", json=json.loads(init_state.json()))
    assert resp.status_code == 200
    assert json.loads(init_state.json()) == json.loads(state.json())


def test_simple_types(state_and_client: Tuple[State, TestClient]):
    state, client = state_and_client
    assert state.lr == float(client.get("/get/lr").text)
    assert state.lr == client.get("/get_from_path?path=lr").json()

    resp = client.put("/set/lr?value=2.0")
    assert resp.status_code == 200
    assert state.lr == 2.0


def test_pydantic_types_full_update(state_and_client: Tuple[State, TestClient]):
    state, client = state_and_client

    assert json.loads(state.checkpoints.json()) == client.get("/get_from_path?path=checkpoints").json()

    resp = client.patch("/set/checkpoints", json={"every_epochs": 3, "save_dir": "other_dir"})
    assert resp.status_code == 200
    assert state.checkpoints.every_epochs == 3
    assert state.checkpoints.save_dir == "other_dir"


def test_pydantic_types_partial_update(state_and_client: Tuple[State, TestClient]):
    state, client = state_and_client

    assert json.loads(state.checkpoints.json()) == client.get("/get_from_path?path=checkpoints").json()

    resp = client.patch("/set/checkpoints", json={"every_epochs": 3})
    assert resp.status_code == 200
    assert state.checkpoints.every_epochs == 3
    assert state.checkpoints.save_dir == "checkpoints"

    # check the update doesn't revert other fields to default values
    resp = client.patch("/set/checkpoints", json={"save_dir": "other_dir"})
    assert resp.status_code == 200
    assert state.checkpoints.every_epochs == 3
    assert state.checkpoints.save_dir == "other_dir"

    # check the inner field update doesn't affect outer fields
    resp = client.patch("/set/model", json={"hidden_dim": [256, 512, 1024]})
    assert resp.status_code == 200
    assert state.model.hidden_dim == [256, 512, 1024]

    assert state.model.head.activation == "relu"
    resp = client.patch("/set/model/head", json={"activation": "sigmoid"})
    assert resp.status_code == 200
    resp = client.put("/set/lr?value=10.")
    assert resp.status_code == 200
    assert state.model.head.activation == "sigmoid"
    assert state.model.hidden_dim == [256, 512, 1024]
    assert state.lr == 10.0
    resp = client.put("/set/model/head/activation?value=silu")
    assert resp.status_code == 200
    assert state.model.head.activation == "silu"
    assert state.model.hidden_dim == [256, 512, 1024]


def test_reset(state_and_client: Tuple[State, TestClient]):
    state, client = state_and_client
    init_state = state.copy(deep=True)
    new_state = State(
        lr=2.0,
        model=Model(hidden_dim=[256, 512, 1024], head=Head(activation="sigmoid")),
    )

    client.patch("/set", json=json.loads(new_state.json()))
    assert state != init_state
    resp = client.delete("/reset")
    assert resp.status_code == 200
    assert init_state == state
