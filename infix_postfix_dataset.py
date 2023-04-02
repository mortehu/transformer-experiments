import lark
import torch
from torch.utils.data import Dataset

import grammar_utils
import to_postfix

grammar = '''
    start: expr
    expr: expr "+" term -> add
        | expr "-" term -> sub
        | term
    term: term "*" factor -> mul
        | term "/" factor -> div
        | factor
    factor: "(" expr ")" -> parens
          | NUMBER
    %import common.NUMBER
    %import common.WS
    %ignore WS
'''

class InfixPostfixDataset(Dataset):
    def __init__(self, num_samples):
        self.num_samples = num_samples
        self.to_postfix = to_postfix.ToPostfix()
        self.parser = lark.Lark(grammar, parser='lalr')
        self.generator = grammar_utils.GrammarRandomGenerator(self.parser)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        infix = self.generator.generate()
        postfix = self.to_postfix.visit(self.parser.parse(infix))[0] + 'E'
        return infix, postfix

class BatchIterator:
    def __init__(self, dataset, batch_size, num_batches):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_batches = num_batches

    def __iter__(self):
        for _ in range(self.num_batches):
            batch = [self.dataset[i % len(self.dataset)] for i in range(self.batch_size)]
            yield collate_fn(batch)

def collate_fn(batch):
    srcs, tgts = zip(*batch)
    src_lens = torch.tensor([len(src) for src in srcs], dtype=torch.long)
    tgt_lens = torch.tensor([len(tgt) for tgt in tgts], dtype=torch.long)
    return srcs, src_lens, tgts, tgt_lens
