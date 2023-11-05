import torch
import numpy as np
from toolbench.inference.ToolDec.clownfish import parser_for_type

class FunctionNameFSM():
    def __init__(self, functions, tokenizer, end_tokens):
        self.function_names = [func["name"] for func in functions]
        self.tokenizer = tokenizer
        self.cur_str = ""
        self.cur_ids = []
        self.end_tokens = end_tokens
        function_trie = []  # todo: implement trie
        for func in self.function_names:
            for i in range(1, len(func)+1):
                function_trie.append(func[:i])
        
        self.trie = set(function_trie)
    
    def __call__(self, logits):
        cand_ids = np.arange(logits.shape[-1])
        if len(self.cur_ids) > 0:
            cur = np.broadcast_to(np.array(self.cur_ids), (cand_ids.shape[0], len(self.cur_ids)))
            prefix_ids = np.concatenate([cur, cand_ids[...,np.newaxis]], axis=-1)
        else:
            prefix_ids = cand_ids[...,np.newaxis]
        prefix = self.tokenizer.batch_decode(prefix_ids)

        if self.cur_str in self.function_names:
            mask = np.isin(cand_ids, self.end_tokens) | np.isin(prefix, list(self.trie))
        else:
            mask = np.isin(prefix, list(self.trie))

        logits[-1, -1, ~mask] = -torch.inf
        return logits

    def push(self, token):
        self.cur_ids.append(token)
        self.cur_str = self.tokenizer.decode(self.cur_ids, skip_special_tokens=True)

class SectionNameFSM():
    def __init__(self, section, tokenizer, end_tokens):
        self.section = section
        self.tokenizer = tokenizer
        self.end_tokens = end_tokens
        section_trie = []
        for i in range(1, len(section)+1):
            section_trie.append(section[:i])
            
        self.trie = set(section_trie)
        self.cur_str = ""
        self.cur_ids = []
    
    def __call__(self, logits):
        cand_ids = np.arange(logits.shape[-1])

        if self.cur_str == self.section:
            mask = np.isin(cand_ids, self.end_tokens)
            logits[-1, -1, ~mask] = -torch.inf
        else:
            if len(self.cur_ids) > 0:
                cur = np.broadcast_to(np.array(self.cur_ids), (cand_ids.shape[0], len(self.cur_ids)))
                prefix_ids = np.concatenate([cur, cand_ids[...,np.newaxis]], axis=-1)
            else:
                prefix_ids = cand_ids[...,np.newaxis]
            prefix = self.tokenizer.batch_decode(prefix_ids)
            mask = np.isin(prefix, list(self.trie))
            mask[0] = False
            logits[-1, -1, ~mask] = -torch.inf
            logits[-1, -1, 0] = -torch.inf


        return logits
    
    def push(self, token):
        self.cur_ids.append(token)
        self.cur_str = self.tokenizer.decode(self.cur_ids, skip_special_tokens=True)

class FunctionInputFSM():
    def __init__(self, tokenizer, end_tokens):
        self.tokenizer = tokenizer
        self.parser = None
        self.end_tokens = end_tokens
        self.cur_ids = []
    
    def __call__(self, logits):

        # Stash the previous state so we can reset back to it if the given token fails
        prev_state = self.parser
        parser = self.parser
        sorted, indices = torch.sort(logits[-1, -1], descending=True)
        print(indices)
        print(sorted)
        for i in indices:
            
        # Iterate through each candidate and set the previously bad ones to be zeroes out
            next = self.tokenizer.decode(i)
            print(f"next:{next}")
            
            next = self.tokenizer.decode(self.cur_ids + [i], skip_special_tokens=True)
            #print("prefix: ", prefix)
            
        # If this is all whitespace and not just a space or the previous token is a space, skip it
            if (next[-1] == " " and next[-2] == " "):
                logits[-1, -1, i] = -float("inf")
                continue
                
            failed = False
            for c in next:
                n = parser.step(parser, c)
                if not n:
                    # print("reject", repr(next), repr(prefix))
                    parser = prev_state
                    failed = True
                    break
                parser = n
            if not failed:
                break
            else:
                logits[-1, -1, i] = -float("inf")

        return logits
    
    def get_parser(self, schema):
        self.parser = parser_for_type(schema, schema)
    
    def push(self, token):
        self.cur_ids.append(token)

class State():
    def __init__(self, fsm, end_tokens, **args):
        if fsm is not None:
            self.fsm = fsm(end_tokens=end_tokens, **args)
        else:
            self.fsm = None
        
        self.end_tokens = end_tokens