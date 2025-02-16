"""
This script is used to convert the provided python code with lite_chains in lang_lite library to actual LangChain code.

change the `langlite_code` variable to the path of the python code to be converted to LangChain code.
`output.py` file will contain the converted code.
"""

from lang_lite import lite_chain
from lang_lite.constants import LLMProvider


def main():
    role = """
            Role: You are an AI assistant who converts provided python code with lite_chains in lang_lite library to actual LangChain code.
            Whole lang_lite library is provided as context and LangChain code syntax are within the lang_lite library.
            Provide output as python code using latest LangChain library.
            """

    try:
        langlite_code = "example.py" # Provide the python code to be converted to LangChain code.

        with open(langlite_code, "r", encoding="utf-8") as file:
            query = file.read()

        converted_code = (lite_chain.SimpleRagChain(LLMProvider.GOOGLE)
                            .add_context_text("example.py")
                            .add_context_text("output_example.py")
                            .add_context_text("setup.py")
                            .add_context_text("README.md")
                            .add_context_directory("./lang_lite", recursive=False, file_pattern="**/*.py")
                            .build_vector_store(LLMProvider.GOOGLE)
                            .prompt_with_embedded_query(query, role))

        # Remove markdown code block markers
        if converted_code.startswith("```python"):
            converted_code = converted_code[len("```python"):].lstrip()
        if converted_code.endswith("```"):
            converted_code = converted_code[:-len("```")].rstrip()

        with open("output.py", "w", encoding="utf-8") as file:
            file.write(converted_code)

        print("Conversion completed successfully. Please check the output.py file for the converted code.")
    except Exception as e:
        print("Error occurred: \n")
        print(e.with_traceback())

if __name__ == "__main__":
    main()