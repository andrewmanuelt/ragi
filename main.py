import sys 
import warnings
warnings.simplefilter("ignore", UserWarning)

from model.komodo import Komodo
from embedding.mpnet import MPNet
from embedding.simcse import SIMCSE
from utility.utility import General
from utility.menu import Menu
from evaluator.evaluator import Evaluator
from hyperparameter.tuning import RetrieverTuning

from utility.loader import Loader

from ray import train, tune

class App(General, Menu):
    def main(self):    
        paramspace = {
            'chunk_size': [300, 500, 700, 1000],
            'chunk_overlap': [0, 30, 50, 60], 
            'top_k': [5, 10]
        }
        
        mpnet = MPNet()
        mpnet.set_params()
        
        em = mpnet.load_embedding()
        emf = mpnet.load_embedding_function()
    
        tuner = RetrieverTuning()
        tuner.set_paramspace(
            paramspace=paramspace
        )        
        tuner.set_configuration(
            persist_dir='./database/single',
            collection='single',
            embedding=em,
            embedding_function=emf,
            test_dir='./dataset/single/single_test.json',
            file_path='./dataset/single/single_train.json'
        )
        tuner.run(context='single')
    
        # self.menu_dataset()
        
        # simsce = SIMCSE()
        # simsce.set_params()
        
        # em = simsce.load_embedding()
        # emf = simsce.load_embedding_function()
        
        # loader = Loader()
        # loader.set_params(
        #     collection='dummy',
        #     embedding=em, 
        #     embedding_function=emf, 
        #     persist_dir='./database/dummy/'
        # )
        
        # # s:exp:1
        # num_k=5
        # score_list=[]
        # result = loader.load_collection().similarity_search_with_relevance_scores(
        #     query=query, k=num_k
        # )
        # for doc, score in result:
        #     print(doc.page_content)
        #     print(score)
        #     score_list.append(score)
        # print(sum(score_list)/num_k)
        # e:exp:1
        
        # s:exp:2
        # retriever = loader.load_collection().as_retriever(search_kwargs={
        #     'k': 5, 
        # })
        # result = retriever.invoke(input=query)
        # print(result)
        # e:exp:2
        

        # komodo = Komodo()
        # model = komodo.model()
        # tokenizer = komodo.tokenizer()
        
        # input = komodo.tokenizing(tokenizer=tokenizer, query=query)
        # result = komodo.generate(model=model, input=input)
        
        # candidate = [
        #     tokenizer.decode(result[0])
        # ]
        
        # reference = [
        #     'Son Gohan'
        # ]
        
        # evaluator = Evaluator()
        # score = evaluator.bert_score(
        #     candidate=candidate,
        #     reference=reference
        # )
        # print(score)
        
        # score = evaluator.rouge(
        #     candidate=candidate,
        #     reference=reference
        # )
        # print(score)
        
        # score = evaluator.meteor(
        #     candidate=candidate,
        #     reference=reference
        # )
        # print(score)

    def menu_dataset(self):
        print('Dataset menu:')
        print("(1) Load single dataset")
        print("(2) Load complex dataset")
        print("(3) Load covid dataset")
        print("(4) Load berita dataset")
        print("(0) Load dummy")
        
        try:
            simcse = SIMCSE()
            embedding = simcse.load_embedding()
            embedding_fn = simcse.load_embedding_function()
            
            menu = int(input('Menu: '))
                
            self.menu_handler(menu, embedding=embedding, embedding_fn=embedding_fn)
        except Exception as e:
            self.errors(message='menu error', e=e)

if __name__ == '__main__':
    try: 
        app = App()
        app.main()
    except KeyboardInterrupt:
        print('Bye...')
        sys.exit(1)


 
    
    