import os 
import sys
import json 
import pandas as pd 

from abc import ABC, abstractmethod

from utility.loader import Loader

class RetrieverAbstract(ABC):
    def __init__(self) -> None:
        self._retriever = None
        self._paramspace = None 
        self._embedding = None 
        self._embedding_function = None 
        self._collection = None 
        self._persist_dir = None 
        self._file_path = None
        self._test_dir = None
        
        if not os.path.exists('./results'):
            os.makedirs('./results', exist_ok=True)
    
    @property 
    def param(self):
        return self._param
    
    @param.setter
    def param(self, param = None):
        self._param = param
    
    @abstractmethod
    def get_param(self):
        pass

    @abstractmethod
    def _load_test(self, dir):
        pass

    @abstractmethod
    def set_configuration(self, embedding, embedding_function, collection, persist_dir):
        pass

class RetrieverTuning(RetrieverAbstract):
    def __init__(self) -> None:
        super().__init__()
        
    def run(self, context):
        df = pd.DataFrame(columns=['chunk_size', 'chunk_overlap', 'top_k', 'mean_score'])
        
        for chunk_size in self._paramspace['chunk_size']:
            for chunk_overlap in self._paramspace['chunk_overlap']:
                for k in self._paramspace['top_k']:
                    df = self._processing(
                        df=df,
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                        k=k
                    )
        
        df.to_excel(f"./results/report_{context}.xlsx")
    
    def _processing(self, df, chunk_size, chunk_overlap, k):
        collection, persist_dir = self._store(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            file_path=self._file_path,
            k=k
        )
                    
        loader = Loader()
        loader.set_params(
            embedding=self._embedding,
            embedding_function=self._embedding_function,
            collection=collection, 
            persist_dir=persist_dir
        )
                    
        retriever = loader.load_collection()
                    
        score = self._count_mean_score(
            retriever=retriever, 
            collection_name=collection,
            k=k, 
        )
                    
        df.loc[len(df)] = {
            'chunk_size': chunk_size,
            'chunk_overlap': chunk_overlap,
            'top_k': k,
            'mean_score': score
        }
                    
        return df
                    
    def _count_mean_score(self, retriever, collection_name, k):
        test = self._load_test()
        
        collection = {
            'item': list(),
            'score': 0
        }
        
        score_list = []
        for doc in test:
            question_collection = {
                'question': doc, 
                'answers': list(),
                'mean_score': 0
            }
            
            score_per_query_list = []
            results = retriever.similarity_search_with_relevance_scores(
                query=doc, k=k
            )
                
            for content, score in results:
                row = {
                    'answer': {
                        'content': content.page_content,
                        'score': score,
                    }    
                }
    
                question_collection['answers'].append(row)
                
                score_per_query_list.append(score)
            
            score_per_query = sum(score_per_query_list)/len(results)
            score_list.append(score_per_query)

            question_collection['mean_score'] = score_per_query
            collection['item'].append(question_collection)
            
        score = sum(score_list) / len(score_list)
        
        collection['score'] = score
        
        with open(f"./results/{collection_name}.json", 'w') as f:
            json.dump(collection, f, indent=4)
            
        return score

    def set_paramspace(self, paramspace):
        self._paramspace = paramspace

    def get_param(self):
        return self._param 
    
    def _store(self, chunk_size, chunk_overlap, k, file_path):
        collection = f"{self._collection}_{chunk_size}_{chunk_overlap}_{k}"
        persist_dir = f"./database/{collection}"
        
        print(f"creating collection {collection} from directory {self._persist_dir}")
        
        loader = Loader()
        loader.set_params(
            embedding=self._embedding, 
            embedding_function=self._embedding_function, 
            collection=collection, 
            persist_dir=persist_dir
        )
        
        loader.store(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            file_path=file_path
        )
        
        return collection, persist_dir    
    
    def set_configuration(self, embedding, embedding_function, collection, persist_dir, test_dir, file_path):
        self._embedding=embedding
        self._embedding_function=embedding_function
        self._collection=collection 
        self._persist_dir=persist_dir
        self._test_dir=test_dir
        self._file_path=file_path

    def _load_test(self) -> list:
        collection = list() 
        data = None 
        
        with open(self._test_dir) as f:
            data = json.load(f)
        
        for row in data:
            collection.append(row['question'])
            
        return collection