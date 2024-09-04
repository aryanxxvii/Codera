from config import *

from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain.text_splitter import Language


def detect_language(document):
    """
    Determines the language of the code in the file.
    """

    class detect(BaseModel):
        """
        Chooses the language of the code in the file.
        """

        language: str = Field(description="Primary programming language used in the code")
    
    
    model = OllamaLLM(model=MODEL_NAME)
    structured_llm = model.with_structured_output(detect)

    template = """Analyze the given code sample and choose the primary programming language used in the code.
    Here is the code: {context}
    The output must be a single word from the list: ['PYTHON', 'JAVASCRIPT', 'CPP', 'JAVA']"""

    prompt = PromptTemplate.from_template(template)

    chain = prompt | structured_llm

    language = chain.invoke({"context": document})

    match(language):
        case "PYTHON":
            return Language.PYTHON
        case "JAVASCRIPT":
            return Language.JS
        case "CPP":
            return Language.CPP
        case "JAVA":
            return Language.JAVA
        case _:
            return Language.MARKDOWN