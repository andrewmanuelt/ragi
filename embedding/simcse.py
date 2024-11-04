from embedding.embedding import EmbeddingAbstract

class SIMCSE(EmbeddingAbstract):
    def __init__(self) -> None:
        super().__init__()
        
        self._embedding_name = 'LazarusNLP/simcse-indobert-base'
        self.set_params()
        
        print(f"embedding name: {self._embedding_name}, device: {self._device}")