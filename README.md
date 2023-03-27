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

The `state` object will now be automatically served over HTTP.

Freak generates `/get/<field>` and `/set/<field>` endpoints for each field in the model, as well as the following endpoints for the root state object:
 - `/get` (`GET`)
 - `/set` (`PATCH`)
 - `/reset` (`DELETE`)
 - `/get_from_path` (`GET`) - which allows to get a value from the state using dot-notation (like `my.inner.field.`)

The `foo` field can now be modified externally by sending a PUT request to the Freak server, which has been automatically started in the background:

```shell
curl -X PUT localhost:4444/set/foo?value=baz
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
    baz: str = "qux"

class State(BaseModel):
    bar: Bar = Bar()
```

Freak will generate `put` endpoints for the `foo` and `baz` fields, and a `patch` endpoint for the `bar` field (as it's a `pydantic` model itself). This `patch` endpoint supports partial updates:

```shell
curl -X PATCH localhost:4444/set/bar -d '{"foo": "baz"}'
```

Because Freak is using `FastAPI`, it's possible to use auto-generated documentation to interact with the Freak server. The interactive documentation can be accessed at Freak's main endpoint, which by default is `localhost:4444`.

The following screenshot shows the generated endpoints for the DL [example](https://github.com/danielgafni/freak/blob/master/examples/dl_example.py):

![Sample Generated Docs](https://raw.githubusercontent.com/danielgafni/freak/master/resources/swagger.png)

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
