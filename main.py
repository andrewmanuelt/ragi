import sys 

from model.gpt2 import GPT2
from embedding.mpnet import MPNet
from embedding.simcse import SIMCSE
from utility.utility import General
from utility.menu import Menu
from evaluator.evaluator import Evaluator

class App(General, Menu):
    def main(self):    
        # self.menu_dataset()
        
        query = "What kind of child is Goku?"

        gpt = GPT2()
        model = gpt.model()
        tokenizer = gpt.tokenizer()
        
        input = gpt.tokenizing(tokenizer=tokenizer, query=query)
        result = gpt.generate(model=model, input=input)
        
        candidate = [
            tokenizer.decode(result[0])
        ]
        
        reference = [
            'Son Gohan'
        ]
        
        evaluator = Evaluator()
        score = evaluator.bert_score(
            candidate=candidate,
            reference=reference
        )
        print(score)
        
        score = evaluator.rouge(
            candidate=candidate,
            reference=reference
        )
        print(score)
        
        score = evaluator.meteor(
            candidate=candidate,
            reference=reference
        )
        print(score)

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
            self.errors(message='Input menu error', e=e)

if __name__ == '__main__':
    try: 
        app = App()
        app.main()
    except KeyboardInterrupt:
        print('Bye...')
        sys.exit(1)


 
    
    