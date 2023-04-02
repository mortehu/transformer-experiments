import lark

class ToPostfix(lark.visitors.Interpreter):
    """Convert infix expressions to postfix expressions."""

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
        """Visit the tree and convert infix to postfix.

        Args:
            tree (lark.Tree): The tree to visit.

        Returns:
            str: The postfix expression.
        """
        if isinstance(tree, lark.Tree):
            return super().visit(tree)
        elif isinstance(tree, lark.Token) and tree.type == 'NUMBER':
            return self.NUMBER(tree)
        else:
            return tree