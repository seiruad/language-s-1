from helpers.strings_with_arrows import string_with_arrows

####################
# Const
####################

DIGITS = '0123456789'

####################
# Position
####################

class Position:
    def __init__(self, idx, ln, col, fn, ftxt):
        self.idx = idx
        self.ln = ln
        self.col = col
        self.fn = fn
        self.ftxt = ftxt

    def advance(self, current_char=None):
        self.idx += 1
        self.col += 1

        if current_char == '\n':
            self.ln += 1
            self.col = 0  

        return self

    def copy(self):
        return Position(self.idx, self.ln, self.col, self.fn, self.ftxt)


####################
# Error
####################

class Error:
    def __init__(self, pos_start: Position, pos_end: Position, err_name: str, details):
        self.pos_start = pos_start
        self.pos_end = pos_end
        self.err_name = err_name
        self.details = details

    def as_string(self):
        result = f'{self.err_name}: {self.details}\n'
        result += f'File {self.pos_start.fn}, line {self.pos_start.ln + 1}\n'
        result += f'\n{string_with_arrows(self.pos_start.ftxt, self.pos_start, self.pos_end)}'
        return result

class IllegalCharError(Error):
    def __init__(self, pos_start: Position, pos_end: Position, details):
        super().__init__(pos_start, pos_end, 'Illegal Character', details)

class InvalidSyntaxError(Error):
    def __init__(self, pos_start: Position, pos_end: Position, details=''):
        super().__init__(pos_start, pos_end, 'Invalid Syntax', details)

class InternalInterpreterException(Exception):
    def __init__(self, err_name: str, details):
        self.err_name = err_name
        self.details = details

    def as_string(self):
        result = f'[Internal Interpreter Exception] {self.err_name}: {self.details}'
        return result


####################
# Tokens
####################

TT_INT =    'INT'
TT_FLOAT =  'FLOAT'
TT_PLUS =   'PLUS'
TT_MINUS =  'MINUS'
TT_MUL =    'MUL'
TT_DIV =    'DIV'
TT_LPAREN = 'LPAREN'
TT_RPAREN = 'RPAREN'
TT_EOF	  = 'EOF'

class Token:
    def __init__(self, type, value=None, pos_start: Position | None = None, pos_end: Position | None=None):
        self.type = type
        self.value = value

        if pos_start:
            self.pos_start = pos_start.copy()
            self.pos_end = pos_start.copy()
            self.pos_end.advance()

        if pos_end:
            self.pos_end = pos_end.copy()
    
    def __repr__(self):
        if self.value:
            return f'{self.type}:{self.value}'
        return f'{self.type}'

####################
# Lexer
####################

class Lexer:
    def __init__(self, fn, text):
        self.fn = fn
        self.text = text
        self.pos = Position(-1, 0, -1, fn, text)
        self.current_char = None
        self.advance()
        
    def advance(self):
        self.pos.advance(self.current_char)
        self.current_char = self.text[self.pos.idx] if self.pos.idx < len(self.text) else None

    def make_tokens(self):
        tokens = []

        while self.current_char != None:

            if self.current_char in ' \t':
                self.advance()
            elif self.current_char in DIGITS:
                tokens.append(self.make_number())
                # self.advance()
            elif self.current_char == '+':
                tokens.append(Token(TT_PLUS, pos_start=self.pos))
                self.advance()
            elif self.current_char == '-':
                tokens.append(Token(TT_MINUS, pos_start=self.pos))
                self.advance()
            elif self.current_char == '*':
                tokens.append(Token(TT_MUL, pos_start=self.pos))
                self.advance()
            elif self.current_char == '/':
                tokens.append(Token(TT_DIV, pos_start=self.pos))
                self.advance()
            elif self.current_char == '(':
                tokens.append(Token(TT_LPAREN, pos_start=self.pos))
                self.advance()
            elif self.current_char == ')':
                tokens.append(Token(TT_RPAREN, pos_start=self.pos))
                self.advance() 
            else:
                pos_start = self.pos.copy()
                char = self.current_char
                self.advance()
                return [], IllegalCharError(pos_start, self.pos, f'"{char}"')
            
        tokens.append(Token(TT_EOF, pos_start=self.pos))
        return tokens, None

    def make_number(self) -> Token:
        num_str = ''
        dot_count = 0
        pos_start = self.pos.copy()

        while self.current_char != None and self.current_char in DIGITS + '.':
            if self.current_char == '.':
                if dot_count == 1:
                    break
                dot_count += 1
                num_str += '.'
            else:
                num_str += self.current_char

            self.advance()

        if dot_count == 0:
            return Token(TT_INT, int(num_str), pos_start, self.pos)
        else:
            return Token(TT_FLOAT, float(num_str), pos_start, self.pos)

####################
# Run
####################

class NumberNode:
    def __init__(self, tok: Token):
        self.tok = tok

    def __repr__(self):
        return f'{self.tok}'

class BinOpNode:
    def __init__(self, left_node, op_tok: Token, right_node):
        self.left_node = left_node
        self.op_tok = op_tok
        self.right_node = right_node

    def __repr__(self):
        return f'({self.left_node}, {self.op_tok}, {self.right_node})'

####################
# Parse results
####################

class ParseResult:
    def __init__(self) -> None:
        self.err = None
        self.node = None

    def register(self, res):
        if isinstance(res, ParseResult):
            if res.err:
                self.err = res.err
            return res.node

        return res


    def success(self, node):
        self.node = node
        return self
    
    def failure(self, err):
        self.err = err
        return self



####################
# Parser
####################

class Parser:
    def __init__(self, tokens: list[Token]) -> None:
        self.tokens = tokens
        self.tok_idx = -1
        self.advance()

    def advance(self) -> Token:
        self.tok_idx += 1
        if self.tok_idx < len(self.tokens):
            self.current_tok = self.tokens[self.tok_idx]
        elif not hasattr(self, 'current_tok'):
            raise InternalInterpreterException(f'Current token is none in Parser', f'self.current_tok must exist but it is not')
        return self.current_tok

    def parse(self):
        res = self.expr()
        if not res.err and self.current_tok.type != TT_EOF:
            return res.failure(
                InvalidSyntaxError(
                    self.current_tok.pos_start, self.current_tok.pos_end, "Expected '+', '-', '*' or '/'"
                )
            )

        return res

    def factor(self):
        res = ParseResult()
        tok = self.current_tok
        if tok.type in (TT_INT, TT_FLOAT):
            res.register(self.advance())
            return res.success(NumberNode(tok))

        return res.failure(
            InvalidSyntaxError(
                tok.pos_start, tok.pos_end, 'Expected INT or FLOAT'
            )
        )

        # raise InternalInterpreterException(f'Factor is not correct', f'self.current_tok[{self.current_tok}] needs to be in (TT_INT, TT_FLOAT) but its not')
        

    def term(self):
        return self.bin_op(self.factor, (TT_MUL, TT_DIV))

    def expr(self):
        return self.bin_op(self.term, (TT_PLUS, TT_MINUS))


    def bin_op(self, func, ops):
        res: ParseResult = ParseResult()
        left = res.register(func())
        if res.err:
            return res

        while self.current_tok.type in ops:
            op_tok = self.current_tok
            res.register(self.advance())
            right = res.register(func())
            left = BinOpNode(left, op_tok, right)

        return res.success(left)



####################
# Run
####################

def run(fn, text):
    # Generate tokens
    lexer = Lexer(fn, text)
    tokens, err = lexer.make_tokens()
    if err:
        return None, err

    # Generate AST
    parser = Parser(tokens)
    ast = parser.parse()

    return ast.node, ast.err
