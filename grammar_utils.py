import collections
import random
import lark

class GrammarRandomGenerator:
    """Randomly generates strings based on a Lark grammar."""

    def __init__(self, parser, max_depth=2):
        """Initialize the GrammarRandomGenerator.

        Args:
            parser (Lark): A Lark parser with the grammar rules.
            max_depth (int, optional): The maximum depth for the generator. Defaults to 2.
        """
        self.rules = collections.defaultdict(list)
        for rule in parser.rules:
            self.rules[rule.origin].append(rule.expansion)
        self.max_depth = max_depth

    def terminal_to_value(self, terminal_name):
        """Convert terminal names to actual characters.

        Args:
            terminal_name (str): The terminal name.

        Returns:
            str: The corresponding character.
        """
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
        """Generate a random string based on the grammar rules.

        Args:
            symbol (lark.grammar.NonTerminal, optional): The starting symbol. Defaults to 'start'.
            depth (int, optional): The current depth of the generation. Defaults to 0.

        Returns:
            str: The generated string.
        """
        if symbol not in self.rules:
            return symbol

        if depth >= self.max_depth:
            alternatives = []
            for rule in self.rules[symbol]:
                if all(child.is_term for child in rule):
                    alternatives.append(rule)
            if not alternatives:
                alternatives = self.rules[symbol]
                alternatives.sort(key=len)
                alternatives = alternatives[:1]
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
