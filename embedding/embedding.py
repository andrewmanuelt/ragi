import torch
import warnings

from abc import ABC, abstractmethod

from langchain_huggingface import HuggingFaceEmbeddings
from chromadb.utils import embedding_functions

warnings.filterwarnings("ignore", category=DeprecationWarning, module="langchain")

class EmbeddingAbstract(ABC):
    def __init__(self) -> None:
        self._embedding_name = ''
        self._device = None 
    
    @property
    def embedding_name(self):
        return self._embedding_name
    
    @property
    def device(self):
        return self._device
    
    @embedding_name.setter
    def embedding_name(self, value):
        self._embedding_name = value 
    
    @device.setter 
    def device(self, value):
        self._device = value
        
    @abstractmethod
    def load_embedding(self) -> any:
        pass 

    @abstractmethod
    def load_embedding_function(self) -> any:
        pass 
    
    def set_params(self):
        self._device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    def load_embedding(self) -> any:
        return HuggingFaceEmbeddings(
            model_name = self._embedding_name, 
            model_kwargs = {
                'device': self._device
            },
        )

    def load_embedding_function(self) -> any:
        return embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name = self._embedding_name
        )



