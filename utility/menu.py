import sys 
import glob

from utility.loader import Loader

class Menu():
    def __init__(self) -> None:
        pass

    def menu_handler(self, menu: int, embedding, embedding_fn):
        if menu == 1:
            self._menu_load_single_dataset(embedding, embedding_fn)
        elif menu == 2:
            self._menu_load_complex_dataset(embedding, embedding_fn)
        elif menu == 3:
            self._menu_load_covid_dataset(embedding, embedding_fn)
        elif menu == 4:
            self._menu_load_berita_dataset(embedding, embedding_fn)
        elif menu == 0:
            self._menu_load_dummy_dataset(embedding, embedding_fn)
        else:
            print('menu not available. exit...')
            sys.exit(1)
        
    def _menu_load_dataset(self, embedding, embedding_fn, collection):
        if collection == 'dummy':
            store_path = str(f"./database/dummy").lower()
            file_path = str(f"./dataset/dummy.json").lower()
        else:
            store_path = str(f"./database/{collection}").lower()
            file_path = str(f"./dataset/{collection}/{collection}_train.json").lower()
        
        loader = Loader()
        
        loader.set_params(
            embedding=embedding,
            embedding_function=embedding_fn,
            collection=collection,
            persist_dir=store_path
        )
        
        loader.store(
            file_path=file_path,
            chunk_size=200,
            chunk_overlap=20
        )
        
    def _menu_load_single_dataset(self, embedding, embedding_fn):
        self._menu_load_dataset(embedding, embedding_fn, 'single')

    def _menu_load_complex_dataset(self, embedding, embedding_fn):
        self._menu_load_dataset(embedding, embedding_fn, 'complex')
    
    def _menu_load_covid_dataset(self, embedding, embedding_fn):
        train = glob.glob('./dataset/covid/covid_train_*.json')
        
        for index, train_d in enumerate(train):
            print(f"processing covid{index}")
        
            collection = f"covid{index}"
            persist_d = f"./database/{collection}"
            
            loader = Loader()
            
            loader.set_params(
                collection=collection,
                embedding=embedding,
                embedding_function=embedding_fn,
                persist_dir=persist_d
            )
            
            loader.store(
                chunk_size=200,
                chunk_overlap=20,
                file_path=train_d
            )

    def _menu_load_berita_dataset(self, embedding, embedding_fn):
        self._menu_load_dataset(embedding, embedding_fn, 'berita')
        
    def _menu_load_dummy_dataset(self, embedding, embedding_fn):
        self._menu_load_dataset(embedding, embedding_fn, 'dummy')
    
        