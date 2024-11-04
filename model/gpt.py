from model.model import LLMAbstract
from utility.utility import General

class GPT(LLMAbstract, General):
    def __init__(self) -> None:
        super().__init__()
        
    def print_params(self):
        print(f"""model name: {self._model_name}, model repo: {self._model_repo}, max token: {self._max_token}, device: {self._device}, temperature: {self._temperature}, repetition penalty: {self.rep_penalty}""")

    def model(self) -> any:
        from transformers import GPT2LMHeadModel
            
        model = GPT2LMHeadModel.from_pretrained(self._model_repo)
        model.generation_config.pad_token_id = model.generation_config.eos_token_id
        
        return model

    def tokenizer(self) -> any:
        from transformers import GPT2TokenizerFast

        tokenizer = GPT2TokenizerFast.from_pretrained(self.model_name)
        
        return tokenizer

    def tokenizing(self, tokenizer, query: str) -> any:
        return tokenizer(query, return_tensors='pt')

    def generate(self, model, input) -> any:
        return model.generate(
            **input, 
            max_new_tokens = self._max_token,
            temperature = self._temperature, 
            repetition_penalty = self._rep_penalty,
            do_sample = True
        )