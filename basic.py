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
# Nodes
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

class UnaryOpNode:
    def __init__(self, op_tok: Token, node) -> None:
        self.op_tok = op_tok
        self.node = node

    def __repr__(self) -> str:
        return f'({self.op_tok}, {self.node})'

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
            self.current_tok: Token = self.tokens[self.tok_idx]
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

        if tok.type in (TT_PLUS, TT_MINUS):
            res.register(self.advance())
            factor = res.register(self.factor())
            if res.err: return res
            return res.success(UnaryOpNode(tok, factor)) 

        elif tok.type in (TT_INT, TT_FLOAT):
            res.register(self.advance())
            return res.success(NumberNode(tok))

        elif tok.type == TT_LPAREN:
            res.register(self.advance())
            expr = res.register(self.expr())
            if res.err: return res
            if self.current_tok.type == TT_RPAREN:
                res.register(self.advance())
                return res.success(expr)
            else:
                return res.failure(
                    InvalidSyntaxError(
                        self.current_tok.pos_start, self.current_tok.pos_end, "Expected ')'"
                    )
                )

        return res.failure(
            InvalidSyntaxError(
                tok.pos_start, tok.pos_end, 'Expected INT or FLOAT'
            )
        )

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
# Values
####################

class Number:
    def __init__(self, value) -> None:
        self.value = value

    def set_pos(self, pos_start: Position | None=None, pos_end: Position | None=None):
        self.pos_start = pos_start
        self.pos_end = pos_end

    def added_to(self, other):
        if isinstance(other, Number):
            return Number(self.value + other.value)  

    def sub_by(self, other):
        if isinstance(other, Number):
            return Number(self.value - other.value) 

    def mul_by(self, other):
        if isinstance(other, Number):
            return Number(self.value * other.value) 
    
    def div_by(self, other):
        if isinstance(other, Number):
            return Number(self.value / other.value)

    def __repr__(self) -> str:
        return str(self.value)


####################
# Interpreter
####################

class Interpreter:
    def visit(self, node):
        method_name = f'visit_{type(node).__name__}'
        method = getattr(self, method_name, self.no_visit_method)
        return method(node)

    def no_visit_method(self, node):
        raise Exception(f'No visit_{type(node).__name__} method defined')

    def visit_NumberNode(self, node):
        # print("visited Number Node")
        return Number(node.tok.value).set_pos(node.pos_start, node.pos_end)
        
    def visit_BinOpNode(self, node):
        # print("visited BinOp Node")
        left: Number = self.visit(node.left_node)
        right = self.visit(node.right_node)

        result = None

        if node.op_tok.type == TT_PLUS:
            result = left.added_to(right)
        elif node.op_tok.type == TT_MINUS:
            result = left.sub_by(right)
        elif node.op_tok.type == TT_MUL:
            result = left.mul_by(right)         
        elif node.op_tok.type == TT_DIV:
            result = left.div_by(right)   

        return result                  
    
    def visit_UnaryOpNode(self, node):
        print("visited UnaryOp Node")
        self.visit(node.node)


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
    if ast.err: return None, ast.err

    # Run program 
    interpreter = Interpreter()
    interpreter.visit(ast.node)

    # return ast.node, ast.err
    return None, None
