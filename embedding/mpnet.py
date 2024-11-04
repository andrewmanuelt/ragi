from embedding.embedding import EmbeddingAbstract

class MPNet(EmbeddingAbstract):
    def __init__(self) -> None:
        super().__init__()
        
        self._embedding_name = 'all-MiniLM-L6-v2'
        self.set_params()
        
        print(f"embedding name: {self._embedding_name}, device: {self._device}")