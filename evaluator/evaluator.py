import inspect
import evaluate 
from evaluate import load 

from nltk.translate.meteor_score import meteor_score
from utility.utility import General

class Evaluator(General):
    def __init__(self) -> None:
        pass
        
    def meteor(self, candidate: list = None, reference: list = None):
        func = self.func_name()
        
        if not isinstance(candidate, list):
            return self.errors(func=func)
        
        if not isinstance(reference, list):
            return self.errors(func=func)

        if len(candidate) != len(reference):
            raise Exception(f"Candidate {len(candidate)} dan reference {len(reference)} length is not same")

        length = len(candidate)
        
        collection = []
        
        for i in range(0, length):
            tok_candidate = str(candidate[i]).split()
            tok_reference = str(reference[i]).split()
            
            score = round(meteor_score([tok_candidate], tok_reference), 4)
            collection.append(score)
        
        return sum(collection) / length
    
    def rouge(self, candidate: list = None, reference: list = None):
        func = self.func_name()
        
        if not isinstance(candidate, list):
            return self.errors(func=func)
        
        if not isinstance(reference, list):
            return self.errors(func=func)

        if len(candidate) != len(reference):
            raise Exception(f"Candidate {len(candidate)} dan reference {len(reference)} length is not same")
        
        rougescore = evaluate.load('rouge')
        
        return rougescore.compute(
            predictions=candidate, 
            references=reference, 
        )
        
    def bert_score(self, candidate: list, reference: list) -> int:
        func = self.func_name()
        
        if not isinstance(candidate, list):
            return self.errors(func=func)
        
        if not isinstance(reference, list):
            return self.errors(func=func)

        if len(candidate) != len(reference):
            raise Exception(f"Candidate {len(candidate)} dan reference {len(reference)} length is not same")
        
        bertscore = load('bertscore')
        
        return bertscore.compute(
            predictions=candidate, 
            references=reference, 
            lang='id'
        ) 
         
    def func_name(self) -> str:
        return inspect.stack()[1].function