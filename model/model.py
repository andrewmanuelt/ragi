import torch

from abc import ABC, abstractmethod
    
class LLMAbstract(ABC):
    def __init__(self) -> None:
        self._model_name = ''
        self._model_repo = ''
        self._max_token = 200
        self._temperature = 0.2
        self._rep_penalty = 1.2
        self._device = None
        self._model = None
    
    @property 
    def model_name(self):
        return self._model_name
    
    @property
    def max_token(self):
        return self._max_token
    
    @property
    def temperature(self):
        return self._temperature
    
    @property
    def rep_penalty(self):
        return self._rep_penalty
    
    @property
    def device(self):
        return self._device

    @property
    def model(self): 
        return self._model
    
    @property 
    def model_repo(self):
        return self._model_repo

    @model_name.setter
    def model_name(self, value):
        self._model_name = value
        
    @max_token.setter
    def max_token(self, value):
        self._max_token = value
        
    @temperature.setter
    def temperature(self, value):
        self._temperature = value
        
    @rep_penalty.setter 
    def rep_penalty(self, value):
        self._rep_penalty = value
    
    @device.setter
    def device(self, value):
        self._device = value
        
    @model_repo.setter
    def model_repo(self, value):
        self._model_repo = value
    
    def set_params(self, model_name: str, model_repo: str, max_token: int, temperature: float, rep_penalty: float):
        self._model_name = model_name
        self._model_repo = model_repo
        self._max_token = max_token
        self._temperature = temperature
        self._rep_penalty = rep_penalty
        self._device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    @abstractmethod
    def model(self) -> any:
        pass
    
    @abstractmethod
    def tokenizer(self) -> any:
        pass
    
    @abstractmethod
    def tokenizing(self, tokenizer) -> any:
        pass 
    
    @abstractmethod
    def generate(self, model, tokenizer) -> any:
        pass 
    
    @abstractmethod
    def print_params(self):
        print(self._model_name)