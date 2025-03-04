# Langchain Lite Library

Langchain Lite is a utility library designed to facilitate faster prototyping with Langchain. It provides tools for working with LLMs, vector stores, and other common needs.

## Installation
To use this library, download the files from `downloads` directory and place them in your project directory.
Open the terminal and navigate to the project directory and run the following command:

```sh
pip install -U lang_lite-<version>-py3-none-any.whl
```


To **compile the wheel file**, install the dependencies and build the package using the following commands.
(**Note:** Using a virtual environment is recommended. For more information, refer to the [Python documentation](https://docs.python.org/3/library/venv.html)) :

### Dependencies:

```sh
pip install -U build
pip install -U setuptools
pip install -U wheel
```

### Build the package:

```sh
python -m build
```

This will generate a `.whl` file in the `dist` directory. Then, you can install the wheel file using `pip`:

```sh
pip install -U dist/lang_lite-<version>-py3-none-any.whl
```

## Usage

Import the library in your code:

```py
import lang_lite
from lang_lite import lite_chain
```

## Documentation
### lite_chain
- `.\lang_lite\lite_chain` is a class that provides a simplified interface to the Langchain API. It is designed to be easy to use and understand, and is ideal for prototyping and testing.
- `Example.py` demonstrates how to use the `lite_chain` class to interact with Langchain.

### langchain_converter
- `.\langchain_converter.py` is a script used to convert Python code that uses `lite_chain` in the `lang_lite` library to actual LangChain code.
- Test your ideas, prototype it and convert it to LangChain code using this script.
- `output_example.py` contains the LangChain code generated from the `Example.py` file.

# For contributors

## Modification

If your modification introduces new dependencies, please add them to `setup.py` under `install_requires` and then rebuild the package.

## Versioning

To update the version of this library:

1. Increment the version number in `setup.py` according to the changes you made (MAJOR, MINOR, or PATCH).
2. Build the new version using `pypa/build`:

    ```sh
    python -m build
    ```

3. Install the updated version using `pip`:

    ```sh
    pip install -U dist/lang_lite-<new_version>-py3-none-any.whl
    ```

## Notes

1. Always use a virtual environment for each project to avoid dependency conflicts.
2. When using a virtual environment, please add the directory to `.gitignore`.
3. Ensure you have the `build` package installed before attempting to build the wheel file.