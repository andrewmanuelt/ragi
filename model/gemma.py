import torch 

from model.model import LLMAbstract
from utility.utility import General

class Gemma(LLMAbstract, General):
    def __init__(self):
        super().__init__()
         
        self.set_device()
        self.set_params()
        self.print_params()

    def model(self) -> any:
        from transformers import AutoModelForCausalLM
        
        model = AutoModelForCausalLM.from_pretrained(self._model_repo)
        model.generation_config.pad_token_id = model.generation_config.eos_token_id
        
        return model 

    def set_device(self):
        self._device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    def tokenizer(self) -> any:
        from transformers import AutoTokenizer
        
        return AutoTokenizer.from_pretrained(self._model_repo, legacy=False)

    def tokenizing(self, tokenizer, query) -> any:
        return tokenizer(query, return_tensors='pt')

    def generate(self, model, input) -> any:
        return model.generate(
            **input, 
            num_beams = 32, 
            early_stopping=True,
            max_new_tokens = self._max_token,
            temperature = self._temperature, 
            repetition_penalty = self._rep_penalty,
            do_sample = True
        )

    def print_params(self):
        print(f"""model name: {self._model_name}, model repo: {self._model_repo}, max token: {self._max_token}, temperature: {self._temperature}, repetition penalty: {self._rep_penalty}""")

    def set_params(self):
        config = self.yamlp()
        config = config['gemma']
        
        self._model_name = config['model_name']
        self._model_repo = config['model_repo']
        self._max_token = config['max_token']
        self._temperature = config['temperature']
        self._rep_penalty = config['repetition_penalty']