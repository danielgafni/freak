import operator
from logging import getLogger
from typing import Any, List, Optional, TypeVar

from fastapi import APIRouter, FastAPI, Query
from pydantic import BaseModel
from starlette.requests import Request
from starlette.responses import JSONResponse
from uvicorn import Config

from freak.uvicorn_threaded import UvicornServer

logger = getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

PATH_QUERY = Query(..., description="Path to field (dot-separated)")


def get_attr_by_path(state: T, path: str) -> Any:
    return operator.attrgetter(path)(state)


def set_attr_by_path(state: T, path: str, value: Any):
    obj = state
    path_parts = path.split(".")
    for p in path_parts[:-1]:
        obj = getattr(obj, p)
    setattr(obj, path_parts[-1], value)


def update_base_model(model: T, update: T):
    """
    Update the model with the update object. The update object can be partial.
    Doesn't copy the model, just updates it in place.
    For some reason pydantic doesn't provide this functionality out of the box.
    :param model:
    :param update:
    :return:
    """
    new_state = type(model)(**model.copy(update=update.dict(exclude_unset=True)).dict())
    for name, field in model:
        setattr(model, name, getattr(new_state, name))


class Freak:
    def __init__(
        self,
        app: Optional[FastAPI] = None,
        prefix: Optional[str] = None,
        host: str = "localhost",
        port: int = 4444,
        docs_url: str = "/",
        uvicorn_log_level: str = "error",
    ):
        """
        Main class for Freak.

        1. Create the Freak instance
        2. Call the control method on a pydantic model
        Optional[3]. Serve the app with the serve method if `control` was called with `serve` set to False

        >>> from freak import Freak
        >>> from pydantic import BaseModel
        >>> class State(BaseModel):
        ...     lr: float = 1e-3
        >>> freak = Freak()
        >>> state = State()
        >>> freak.control(state)  # doctest: +SKIP

        :param app: the FastAPI app to add the routes to. Will create a new one if not provided.
        :param prefix: the prefix to add to the routes
        :param host: the host to serve the app on
        :param port: the port to serve the app on
        :param docs_url: the docs url to use for the FastAPI app
        :param uvicorn_log_level: the log level for the uvicorn server
        """
        self.app = app or FastAPI(title="Freak", description="Application state control", docs_url=docs_url)
        self.prefix = prefix or ""
        self.host = host
        self.port = port
        self.uvicorn_log_level = uvicorn_log_level

    def control(self, state: T, serve: bool = True):
        if not state.Config.allow_mutation:
            state.Config.allow_mutation = True
            logger.warning("Changed allow_mutation to True for the state object because Freak needs it to be mutable")

        state = state

        self.add_routes(self.app, state)

        if serve:
            self.serve()

    def serve(self):
        self.server = UvicornServer(
            config=Config(app=self.app, host=self.host, port=self.port, log_level=self.uvicorn_log_level)
        )
        self.server.run_in_thread()
        # logger.info(f"Running Freak on http://{self.host}:{self.port}")

    def stop(self):
        self.server.cleanup()

    def add_routes(self, app: FastAPI, state: T) -> FastAPI:
        init_state = state.copy(deep=True)

        router = APIRouter(prefix=self.prefix)

        state_name = state.__repr_name__()

        @router.post("/stop", description="Stop the Freak server", tags=["stop"])
        async def stop_server():  # pyright: ignore
            self.stop()

        @router.get("/get", description=f"Get the whole {state_name}", tags=[state_name])
        async def get_state() -> type(state):  # pyright: ignore
            return state

        @router.patch(
            "/set", description=f"Patch the whole {state_name}. Partial updates are supported.", tags=[state_name]
        )
        async def set_state(inpt: type(state)) -> str:  # pyright: ignore
            old_state = state.copy()
            update_base_model(state, inpt)
            logger.info(f"Set the whole state from {old_state} to {inpt}")
            return "success"

        @router.delete("/reset", description=f"Reset the whole {state_name} to initial value", tags=[state_name])
        async def reset_state() -> str:
            old_state = state.copy()  # pyright: ignore[reportOptionalMemberAccess]
            # no ide why do I have to convert the init_state to dict and back to T,
            # but otherwise the reset test fails
            update_base_model(state, type(state)(**init_state.dict()))
            logger.info(f"Reset the whole state from {old_state} to {state}")
            return "success"

        @router.get("/get_from_path", description=f"Get {state_name} parameter by path", tags=[state_name])
        async def get_field_from_path(path: str = PATH_QUERY) -> JSONResponse:
            return get_attr_by_path(state, path)

        def make_get_param_handler(path: str, field: Any):
            async def get_param(request: Request) -> type(field):  # pyright: ignore
                return get_attr_by_path(state, path)

            get_param.__name__ = f"get_{path}"

            return get_param

        def make_set_param_handler(path: str, field: Any):
            if not isinstance(field, BaseModel):

                async def set_param(request: Request, value: type(field)) -> str:  # pyright: ignore
                    set_attr_by_path(state, path, value)
                    logger.info(f"Set {path} to {value}")
                    return "success"

            else:

                async def set_param(request: Request, value: type(field)) -> str:  # pyright: ignore
                    field_to_update = get_attr_by_path(state, path)
                    update_base_model(field_to_update, value)
                    logger.info(f"Patched {path} with {value}")
                    return "success"

            return set_param

        def add_handlers_for_model(model: BaseModel, prefix_key: Optional[List[str]] = None):
            prefix_key = prefix_key or []
            assert isinstance(prefix_key, list)
            for name, field in model:
                full_key = prefix_key + [name]
                prefix_url = "/".join(full_key)
                prefix_path = ".".join(full_key)
                tags = [".".join(full_key)]

                router.get(
                    f"/get/{prefix_url}",
                    tags=tags,  # pyright: ignore[reportGeneralTypeIssues]
                    name="",  # names are being transformed by FastAPI in a weird way
                    description=f"Get {prefix_path}",
                )(make_get_param_handler(prefix_path, field))

                # put will replace the whole object, patch will update it

                if not isinstance(field, BaseModel):
                    desc = f"Set {prefix_path}."
                    method = router.put
                else:
                    desc = f"Patch {prefix_path}. Partial updates are supported."
                    method = router.patch

                method(
                    f"/set/{prefix_url}",
                    tags=tags,  # pyright: ignore[reportGeneralTypeIssues]
                    name="",  # names are being transformed by FastAPI in a weird way
                    description=desc,
                )(make_set_param_handler(prefix_path, field))

                if isinstance(field, BaseModel):
                    add_handlers_for_model(field, full_key)

        add_handlers_for_model(state)

        app.include_router(router)

        return app


def control(
    state: T,
    app: Optional[FastAPI] = None,
    serve: bool = True,
    prefix: Optional[str] = None,
    host: str = "localhost",
    port: int = 4444,
    docs_url: str = "/",
    uvicorn_log_level: str = "error",
) -> Freak:
    """
    Freak's main function.
    Puts the state under Freak's control and adds routes to the app to control the state remotely.
    The `serve` argument can be set to False if you want to serve the app yourself.
    :param state: the state object to control, should be a pydantic BaseModel
    :param app: the FastAPI app to add the routes to. Will create a new one if not provided.
    :param serve: whether to serve the app or not. If False, you will have to serve the app yourself.
    :param prefix: the prefix to add to the routes
    :param host: the host to serve the app on
    :param port: the port to serve the app on
    :param docs_url: the url to serve the docs on for the FastAPI app
    :param uvicorn_log_level: the log level for the uvicorn server
    :return: the Freak object which can be used to serve the app later with `serve` set to `False`

    >>> from freak import control
    >>> from pydantic import BaseModel
    >>> class State(BaseModel):
    ...     a: int = 1
    >>> state = State()
    >>> control(state)  # doctest: +SKIP
    """
    freak = Freak(app=app, prefix=prefix, host=host, port=port, docs_url=docs_url, uvicorn_log_level=uvicorn_log_level)
    freak.control(state, serve=serve)
    return freak
