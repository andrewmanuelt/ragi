import os
import glob
import json
import chromadb

from tqdm import tqdm
from abc import ABC, abstractmethod
from langchain_chroma.vectorstores import Chroma
from langchain_community.document_loaders import JSONLoader

from utility.utility import Splitter

class LoaderAbstract(ABC):
    def __init__(self) -> None:
        self._embedding_fn = None 
        self._embedding = None
        self._collection_name = '' 
        self._persist_directory = ''
    
    @property 
    def embedding_fn(self):
        return self._embedding_fn
    
    @property 
    def embedding(self):
        return self._embedding
    
    @property
    def collection_name(self):
        return self._collection_name
    
    @property
    def persistent_directory(self):
        return self._persist_directory
    
    @embedding_fn.setter
    def embedding_fn(self, value):
        self._embedding_fn = value
        
    @embedding.setter
    def embedding(self, value):
        self._embedding = value
    
    @collection_name.setter
    def collection_name(self, value):
        self._collection_name = value
    
    @persistent_directory.setter
    def persist_directory(self, value):
        self._persist_directory = value
    
    @abstractmethod
    def _metadata_procesing(self, record, metadata):
        pass
    
    @abstractmethod
    def _load_document(self, file_path):
        pass 
    
    @abstractmethod
    def _add_document(self, content, context, counter, splitter):
        pass
    
    @abstractmethod
    def load_collection(self):
        pass 
    
    @abstractmethod
    def store(self, file_path: str):
        pass
    
    def set_params(self, embedding: any, embedding_function: any, collection: str, persist_dir: str):
        self._embedding = embedding
        self._embedding_fn = embedding_function
        self._collection_name = collection
        self._persist_directory = persist_dir

class Loader(LoaderAbstract):
    def __init__(self) -> None:
        super().__init__()
        
    def _persistent_client(self):
        return chromadb.PersistentClient(
            path = self._persist_directory
        )
    
    def check_is_collection_exist(self, collection_name):
        client = self._persistent_client()
        collection_list = client.list_collections()
        
        if collection_name in [collection.name for collection in collection_list]:
            print(f"collection {collection_name} already exist")
            return True
        else:
            return False
        
    def _check_is_collection_exist(self, collection_name):
        client = self._persistent_client()
        collection_list = client.list_collections()
        
        if collection_name in [collection.name for collection in collection_list]:
            print(f"collection {collection_name} already exist")
            return True
        else:
            return False
            
    def _collection_instance(self):
        client = self._persistent_client()
        
        if not (self._check_is_collection_exist(self._collection_name)):
            return client.create_collection(
                    name = self._collection_name,
                    embedding_function = self._embedding_fn,
                    metadata={"hnsw:space": "cosine"}
                )
            
    def _metadata_procesing(self, record, metadata):
        metadata['id'] = record.get('id')
        metadata['context'] = record.get('context')
        
        return metadata

    def load_collection(self):
        return Chroma(
            persist_directory=self._persist_directory, 
            collection_name=self._collection_name,
            embedding_function=self._embedding
        )

    def _load_document(self, file_path):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Cannot find file in path {file_path}")

        loader = JSONLoader(
            file_path = file_path,
            jq_schema = ".[]",
            text_content=False,
            metadata_func=self._metadata_procesing
        )
        
        return loader.load()
        
    def store(self, file_path: str, chunk_size: int, chunk_overlap: int):
        if self._check_is_collection_exist(self._collection_name):
            return True
        
        documents = self._load_document(file_path)
        collection = self._collection_instance()
        
        for document in tqdm(documents, desc='Processing document'):
            self._add_document(
                collection=collection,
                content=document,
                chunk_size=chunk_size, 
                chunk_overlap=chunk_overlap
            )

    def _add_document(self, collection, content, chunk_size, chunk_overlap):
        splitter = Splitter(
            chunk_size=chunk_size, 
            chunk_overlap=chunk_overlap
        )
        
        data = json.loads(content.page_content)
        page_content = data["question"] + " " + data ['answer']
        
        context_split = splitter.call(
            text=page_content,
        )
            
        for i, context in enumerate(context_split):
            collection.add(
                documents=[context],
                metadatas=[{
                    "context": content.metadata['context']
                }],
                ids=["{id}p{part}".format(id=str(content.metadata['id']), part=i)]
            )

    def load_covid(self, chunk_size, chunk_overlap, embedding, embedding_function):
        train = glob.glob('./dataset/covid/covid_train_*.json')
        
        for index, train_d in enumerate(train):
            print(f"processing covid{index}")
        
            collection = f"covid{index}"
            persist_d = f"./database/{collection}"
            
            self.set_params(
                collection=collection,
                embedding=embedding,
                embedding_function=embedding_function,
                persist_dir=persist_d
            )
            
            self.store(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                file_path=train_d
            )