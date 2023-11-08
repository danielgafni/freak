# Freak

Control.

Control your application state with a single line of code.

Freak is using `pydantic` to define the state, supports nested models, partial updates, data validation, and uses `FastAPI` to serve the state over HTTP.

## Installation
```shell
pip install freak
```

## Usage

Define a `pydantic` model and pass it to the `control` function.

```python
from freak import control
from pydantic import BaseModel

class State(BaseModel):
    foo: str = "bar"

state = State()
control(state)
```

The `state` object will now be automatically served over HTTP (in a separate thread).

Freak generates `/get/<field>` and `/set/<field>` endpoints for each field in the model, as well as the following endpoints for the root state object:
 - `/get` (`GET`)
 - `/set` (`PATCH`)
 - `/reset` (`DELETE`)
 - `/get_from_path` (`GET`) - which allows to get a value from the state using dot-notation (like `my.inner.field.`)

The `foo` field can now be modified externally by sending a PUT request to the Freak server, which has been automatically started in the background:

```shell
curl -X PUT localhost:4444/set/foo?value=baz
"success"
```

At the same time, the `state` object cat be used in the program. Freak will always modify it in place. This can be helpful for long-running programs that need to be controlled externally, like:
 - training a neural network
 - running a bot
 - etc.

Freak supports nested models and partial updates. Consider the following model:

```python
from pydantic import BaseModel

class Bar(BaseModel):
    foo: str = "bar"
    baz: int = 0

class State(BaseModel):
    bar: Bar = Bar()
```

Freak will generate `put` endpoints for the `foo` and `baz` fields, and a `patch` endpoint for the `bar` field (as it's a `pydantic` model itself). This `patch` endpoint supports partial updates:

```shell
curl -X 'PATCH' \  
  'http://localhost:4444/set/bar' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{"foo": "baz"}'
"success"
```

`pydantic` will guard the state from wrong types:


```shell
curl -X 'PATCH' \  
  'http://localhost:4444/set/bar' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{"baz": "lol"}'

{"detail":[{"loc":["body","baz"],"msg":"value is not a valid integer","type":"type_error.integer"}]}
```

Because Freak is using `FastAPI`, it's possible to use auto-generated documentation to interact with the Freak server. The interactive documentation can be accessed at Freak's main endpoint, which by default is `localhost:4444`.

A more useful example is the following PyTorch Lightning Callback which stops training when the `state.should_stop` field is set to `True`:

```python
from typing import Optional

import lightning as L
from freak import Freak
from lightning.pytorch.utilities.distributed import rank_zero_only
from pydantic import BaseModel


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

        if self.state.should_stop:  # call the Freak API to set this to True
            # this triggers lightning to stop training
            trainer.should_stop = True
            trainer.strategy.barrier()

    @rank_zero_only
    def on_train_end(self, trainer: "L.Trainer", pl_module: "L.LightningModule") -> None:
        self.freak.stop()
```

### Generated OpenAPI UI

The following screenshot shows the generated endpoints for the [ML example](https://github.com/danielgafni/freak/blob/master/examples/dl_example.py). Warning: making ML pipelines less reproducible by changing hyperparameters on the fly isn't the brightest idea!

![Sample Generated Docs](https://raw.githubusercontent.com/danielgafni/freak/master/resources/swagger.png)

Passing your own FastAPI app as `control(state, app=app)` allows to use Freak in an existing project. The app can be also customized with other endpoints. One of the reasons for doing this might be adding more RPC-like functionality like calling a Python function from the Freak server explicitly.

## Development

### Installation

```shell
poetry install
poetry run pre-commit install
```
### Testing

```shell
poetry run pytest
```
