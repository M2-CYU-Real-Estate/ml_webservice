# ML Webservice 

This is the webservice that hosts Machine Learning components for the `PDS-Real Estate` project.

## Requirements

This project requires python with version at least 3.8

## Initial setup

After cloning the repository, it is advised to create a virtual environment inside it : 

```shell
# On windows
py -m venv env
env\Scripts\activate.bat

# On linux
python3 - venv env
source env/bin/activate
```

This will create a `env` folder containing all the environment stuff, and activate the environment.

> Do NOT change the name of this folder, or else it could be committed to git !

When the `env` environment is active, get the dependencies using pip : 

```shell
pip install -r requirements.txt
```

> If more dependencies are needed, do not forget to update the file : 
> ```shell
> pip freeze > requirements.txt
> ```

If needed (you certainly need for setting paths), After that, go in the `resources` folder and duplicate the `.env.TEMPLATE` file in order to create the `.env` file. You can then update the variables locally : only the variables in the `.env` file will be taken account

> The `.env` file is local and will never be committed. The `.env.TEMPLATE` file should never be modified for personal use, rather when adding or modifying a variable to set.

## Running the server

Before starting the server, assure that you are on the right environment. You can activate it using the following command : 

```shell
# On windows
env\Scripts\activate.bat

# On linux
source env/bin/activate
```

The server could simply be started using the script `server.sh` : 

```shell
# Linux
./server.sh
# Windows
server.bat
```

You can also directly run the command : 

```shell
uvicorn uvicorn app.main:app --reload
```

On both options, you can pass multiple command line, like `--port` in order to define on which port the server should listen to :

```shell
# Linux
./server.sh --port 5555
# Windows
server.bat --port 5555
# OR
uvicorn uvicorn app.main:app --reload --port 5555
```

> By default, the port set is 8000.

> The `--reload` option permits setting up the development version of the server, which will automatically reload the application when saving code.

## How does it work ?

### Some concepts

FastAPI relies heavily on Pydantic, a library that permits to create data classes (similar to what python does with `@dataclass`, but more robust). They are heavily used for defining object request and response bodies.

For any function, only the wanted dependencies are needed : we can provide dependencies directly in other functions with the `Depends` method : 

```python
from fastapi import Depends

@app.get("/some-route")
def some_function(some_file_path: str = Depends(retrieve_some_file_path)):
    pass
```

Here, `retrieve_some_file_path` is a function that will be called directly by the framework in order to create the route function.

> For more complex objects, we can even chain dependencies:

```python
# The '@lru_cache' is set to indicate that the function 
# should only be called once : this creates a singleton dependency !

@lru_cache()
def sub_service() -> SubService:
    return SubService()

@lru_cache()
def main_service(sub_service_obj: SubService = Depends(sub_service)) -> MainService:
    return MainService(sub_service_obj)
```

### Structure

The root of the server is the fie `main.py`, containing all global configuration and route definition.

In each folder under the `app` folder, we have:
 - The `__init__.py` file, mandatory for any python package folder (which permits expressing what can be imported from the package).
 - A `dependencies.py` file, which contains all functions of the module that are designed to be used with the `Depends` method (i.e., to perform dependency injection). The services will then be created here.

> In the `__init__.py` file, you should then have at least a line (along those for importing types and functions), one that exports all functions from the `dependencies` module :
> ```python
> from .dependencies import *
> ```


### Add properties

All the `.env` and `.env.local` file's variables end up in a properties holder named `Properties` (located in `app\config\properties.py`). If you need to add a variable, you can add an attribute to the class directly (respecting the format like the other ones) : 

```python
class Properties(BaseSettings):
    """A configuration class gathering all variables in '.env...' files.

    This supports '.env' file and '.env.local' files.
    """
    api_version: str = "1.0"
    regression_model_path: str
    # <-- ADD ATTRIBUTE HERE
    ...
```
