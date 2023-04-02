import collections
import random
import sys

import lark

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

parser = lark.Lark(grammar, parser='lalr')

class GrammarRandomGenerator:
    def __init__(self, parser, max_depth=3):
        self.rules = collections.defaultdict(list)
        for rule in parser.rules:
            self.rules[rule.origin].append(rule.expansion)
        self.max_depth = max_depth

    def terminal_to_value(self, terminal_name):
        if terminal_name == "PLUS":
            return "+"
        elif terminal_name == "MINUS":
            return "-"
        elif terminal_name == "STAR":
            return "*"
        elif terminal_name == "SLASH":
            return "/"
        elif terminal_name == "LPAR":
            return "("
        elif terminal_name == "RPAR":
            return ")"
        elif terminal_name == "NUMBER":
            return str(random.randint(0, 9))
        else:
            return terminal_name

    def generate(self, symbol=lark.grammar.NonTerminal('start'), depth=0):
        if symbol not in self.rules:
            return symbol

        if depth >= self.max_depth:
            alternatives = []
            for rule in self.rules[symbol]:
                if all(child.is_term for child in rule):
                    alternatives.append(rule)
            if not alternatives:
                alternatives = self.rules[symbol]
        else:
            alternatives = self.rules[symbol]
        chosen_alternative = random.choice(alternatives)


        generated = ''
        for child in chosen_alternative:
            if child.is_term:
                generated += self.terminal_to_value(child.name)
            else:
                generated += self.generate(symbol=child, depth=depth+1)

        return generated


class ToPostfix(lark.visitors.Interpreter):
    def add(self, tree):
        return self.visit(tree.children[0]) + ' ' + self.visit(tree.children[1]) + ' +'

    def sub(self, tree):
        return self.visit(tree.children[0]) + ' ' + self.visit(tree.children[1]) + ' -'

    def mul(self, tree):
        return self.visit(tree.children[0]) + ' ' + self.visit(tree.children[1]) + ' *'

    def div(self, tree):
        return self.visit(tree.children[0]) + ' ' + self.visit(tree.children[1]) + ' /'

    def parens(self, tree):
        return self.visit(tree.children[0])

    def expr(self, tree):
        return self.visit(tree.children[0])

    def term(self, tree):
        return self.visit(tree.children[0])

    def factor(self, tree):
        return self.visit(tree.children[0])

    def NUMBER(self, tree):
        return tree.value

    def visit(self, tree):
        if isinstance(tree, lark.Tree):
            return super().visit(tree)
        elif isinstance(tree, lark.Token) and tree.type == 'NUMBER':
            return self.NUMBER(tree)
        else:
            return tree

# Generate a random expression
generator = GrammarRandomGenerator(parser)
generated_expression = generator.generate()
print("Generated expression:", generated_expression)

tree = parser.parse(generated_expression)
to_postfix = ToPostfix()
postfix = to_postfix.visit(tree)

print("In postfix:", postfix)
