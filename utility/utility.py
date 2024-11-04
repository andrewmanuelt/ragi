import sys
import time
import inspect

from langchain_text_splitters import RecursiveCharacterTextSplitter

from abc import ABC


class Utility(ABC):
    def __init__(self) -> None:
        pass
    
    @property 
    def fn(self):
        return self._fn

    @fn.setter
    def fn(self, value):
        self._fn = value
    
class General(Utility):    
    def _get_function_name(self):
        return inspect.stack()[0].function
    
    def errors(self, func: str = None, e: Exception = None, message: str = None):
        timestamp = time.strftime("%d/%m/%Y %H:%I:%S")
        
        if message is not None:
            e = message 
        
        if func is None and e is None:
            print(f"[{timestamp}][E]")
        elif func is not None:
            print(f"[{timestamp}][E][{func}] {e}")
        elif e is not None:
            print(f"[{timestamp}][E] {e}")
            
        return sys.exit(1)

    def func_name(self) -> any:
        raise NotImplementedError

    def load_prompt_template(self, question, context):
        template = None
        
        with open('./prompt/ReAct_en.txt') as f:
            template = f.read()

        return template.format(question=question, context=context)

    def append_context(self, documents):
        collection = []
        
        print(documents)
        
        for document in documents:
            print(document.page_content)
            sys.exit(1)
        
        return  "".join(collection)
     
class Prompt(Utility):
    def call(self) -> any:
        raise NotImplementedError

class Splitter(Utility):
    def __init__(self, chunk_size, chunk_overlap):
        self._separators = ['\n', '\n\n', '.', '. ', ' . ']
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        
    def call(self, text: str) -> any:
        splitter = RecursiveCharacterTextSplitter(
            separators = self._separators,
            chunk_size = self._chunk_size,
            chunk_overlap = self._chunk_overlap,
            length_function = len,
            is_separator_regex = False,
        )
        
        return splitter.split_text(text)
    
# gen = General(fn='general')
# gen.errors()
