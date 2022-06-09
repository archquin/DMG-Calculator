


###############################
#Imports
###############################

from strings_with_arrows import *
import numpy as np



###############################
#Symbols and variables
###############################


CHARACTERS='ABCDEFGHIJKLMNOPQRSTUWVXYZ-<>/\\~|'
CHARACTER = 'ABCDEFGHIJKLMNOPQRSTUWVXYZ'
DIGITS='0123456789'
SYMBOLS='|~-<>/\\'


###############################
#Errors
###############################

class Error:
    def __init__(self,pos_start,pos_end,error_name,details):
        self.pos_start= pos_start
        self.pos_end=pos_end
        self.error_name=error_name
        self.details=details

    def as_string(self):
        result = f'{self.error_name}:{self.details}'
        result += f'File {self.pos_start.fn}, line {self.pos_start.ln +1}'
        result += '\n\n' + string_with_arrows(self.pos_start.ftxt,self.pos_start,self.pos_end)
        return result


class IllegalDigitError(Error):
    def __init__(self,pos_start,pos_end,details):
        super().__init__(pos_start,pos_end,'Illegal Digit', details)

class InvalidSyntax(Error):
    def __init__(self,pos_start,pos_end,details=''):
        super().__init__(pos_start,pos_end,'Illegal Symbol', details)


###############################
#Position
###############################
class Position:
    def __init__(self,idx,ln,col,fn,ftxt):
        self.idx= idx
        self.ln= ln
        self.col = col
        self.fn = fn
        self.ftxt = ftxt

    def advance(self,current_char= None):
        self.idx +=1
        self.col += 1

        if current_char =='\n':
            self.ln+=1
            self.col =0

    def copy(self):
        return Position(self.idx, self.ln, self.col, self.fn, self.ftxt)

###############################
#Tokens
###############################


TT_STR='LOGIC_VARIABLE'
TT_BOOL='BOOLEAN_VARIABLE'
TT_IF='IF'
TT_IFF='IFF'
TT_AND='AND'
TT_OR='OR'
TT_LPAREN='LPAREN'
TT_RPAREN='RPAREN'
TT_NEGATION='NOT'
TT_ENTAILS='ENTAILS'
TT_ID='IDENTITY'
TT_EOF='EOF'


class Token:
    def __init__(self,type_,value=None,pos_start=None,pos_end=None):
        self.type= type_
        self.value= value

        if pos_start:
            self.pos_start = pos_start.copy()
            self.pos_end = pos_start.copy().advance()

        if pos_end:
            self.pos_end = pos_end

    def __repr__(self):
        #if self.value: return f'{self.type}:{self.value}'
        if self.type in (TT_STR,TT_BOOL,TT_NEGATION): return f'{self.value}'
        return f'{self.type}'


class Lexer:
    def __init__(self,fn,text):
        self.fn = fn
        self.text = text
        self.pos = Position(-1,0,-1, fn, text)
        self.current_char = None
        self.advance()

    def advance(self):
        self.pos.advance(self.current_char)
        self.current_char= self.text[self.pos.idx] if self.pos.idx < len(self.text) else None

    def make_tokens(self):
        tokens =[]

        while self.current_char != None:
            if self.current_char in ' \t':
                self.advance()
            elif self.current_char in CHARACTERS:
                if self.current_char != 'T' and self.current_char !='F' :
                    tokens.append(self.make_character())
                else:
                    tokens.append(self.make_statement())

            elif self.current_char =='(':
                tokens.append(Token(TT_LPAREN,pos_start=self.pos))
                self.advance()
            elif self.current_char ==')':
                tokens.append(Token(TT_RPAREN,pos_start=self.pos))
                self.advance()
            else:
                pos_start=self.pos.copy()
                char = self.current_char
                self.advance()
                return [], IllegalDigitError(pos_start, self.pos ,"'" + char + "'")
        tokens.append(Token(TT_EOF,pos_start=self.pos))
        return tokens, None




    def make_character(self):
        num_str=''
        pos_start=self.pos.copy()

        while self.current_char != None and self.current_char in CHARACTERS:

            num_str+=self.current_char
            self.advance()
        if num_str[0] not in SYMBOLS:
            return Token(TT_STR,str(num_str),pos_start,self.pos)
        else:
            if num_str == '-->':
                return Token(TT_IF,str(num_str),pos_start)
            elif num_str == '<--':
                return Token(TT_IF,str(num_str),pos_start)
            elif num_str == '<-->':
                return Token(TT_IFF,str(num_str),pos_start)
            elif num_str == '\\/':
                return Token(TT_AND,str(num_str),pos_start)
            elif num_str == '/\\':
                return Token(TT_OR,str(num_str),pos_start)
            elif num_str[0] == '~':
                j = 0
                for i in range(0,len(num_str)):
                    j += 1
                if j%2 != 0:
                    num_str='~'
                elif j%2 == 0:
                    num_str='!'
                return Token(TT_NEGATION,str(num_str),pos_start)
            elif num_str == '|--':
                return Token(TT_ENTAILS,str(num_str),pos_start)
            elif num_str == '|||':
                return Token(TT_ID,str(num_str),pos_start)
            else:
                return ('error')




    def make_statement(self):
        num_str=''
        pos_start=self.pos.copy()
        while self.current_char != None and self.current_char in CHARACTERS:
            num_str+=self.current_char
            self.advance()
            if num_str == 'T' :

                return Token(TT_BOOL,bool(True),pos_start,self.pos)
            elif num_str == 'F' :

                return Token(TT_BOOL,str(bool(False)),pos_start,self.pos)

###############################
#Nodes
###############################

class InputNode:
    def __init__(self,tok):
        self.tok = tok

    def __repr__(self):
        return f'{self.tok}'


class Bin0pNode:
    def __init__(self,left_node,op_tok,right_node):
        self.left_node = left_node
        self.op_tok= op_tok
        self.right_node = right_node

    def __repr__(self):
        if self.left_node != None:
            return f'({self.left_node},{self.op_tok},{self.right_node})'
#        else :
#            return f'({self.op_tok},{self.right_node})'


class UnaryOpNode:
    def __init__(self,op_tok,right_node):
        self.op_tok= op_tok
        self.right_node = right_node

    def __repr__(self):
        return f'{self.op_tok}{self.right_node}'





###############################
#Parse results
###############################

class ParseResult:
    def __init__(self):
        self.error = None
        self.node = None

    def register(self,res):
        if isinstance(res,ParseResult):
            if res.error: self.error = res.error
            return res.node
        return res


    def success(self, node):
        self.node = node
        return self

    def failure(self,error):
        self.error = error
        return self


###############################
#Parser
###############################

class Parser:
    def __init__(self,tokens):
        self.tokens = tokens
        self.tok_idx = -1
        self.advance()

    def advance(self, ):
        self.tok_idx += 1
        if self.tok_idx < len(self.tokens):
            self.current_tok = self.tokens[self.tok_idx]
        return self.current_tok

    def previous(self, ):
        self.tok_idx -= 1
        if self.tok_idx >= 0:
            self.current_tok = self.tokens[self.tok_idx]
        return self.current_tok


    def parse(self):
        res=self.expr()
        if not res.error and self.current_tok.type != TT_EOF:
            return res.failure(InvalidSyntax(self.current_tok.pos_start, self.current_tok.pos_end, 'err'))
        return res


    def factor(self):
        res = ParseResult()
        tok = self.current_tok
        if tok.type in (TT_ENTAILS,TT_IF,TT_IFF,TT_AND,TT_OR):
            res.register(self.advance())
            factor=res.register(self.advance())
        #    print(factor,'factor')
            if res.error: return res
            return res.success(Bin0pNode(InputNode(tok),tok,factor))

        elif tok.type in (TT_STR):
            res.register(self.advance())
            return res.success(InputNode(tok))

        elif tok.type == TT_LPAREN:
            self.previous()
            b4 = self.current_tok.value
            if b4 == '~':
        #        print(b4,'b4')
                self.advance()
                res.register(self.advance())
                expr = res.register(self.xpr())

            else:
        #        print(b4)
                self.advance()
                res.register(self.advance())
                expr = res.register(self.expr())

        #    print(expr,'Expr')
            if res.error: return res
            if self.current_tok.type == TT_RPAREN:
                res.register(self.advance())
                return res.success(expr)
            else:
                return res.failure(InvalidSyntax(tok.pos_start,tok.pos_end,'can t find parenthesis'))
        res.failure(InvalidSyntax(tok.pos_start,tok.pos_end,'err'))


    def negation(self):
        return self.un_op(self.factor, (TT_NEGATION))

    def AND(self):
        return self.bin_op(self.negation,(TT_AND))

    def OR(self):
        return self.bin_op(self.AND,(TT_OR))

    def expr(self):
        return self.bin_op(self.OR,(TT_ENTAILS))

    def xnegation(self):
        return self.un_op(self.factor, (TT_NEGATION))

    def xOR(self):
        return self.bin_op(self.xnegation,(TT_OR))

    def xAND(self):
        return self.bin_op(self.xOR,(TT_AND))

    def xpr(self):
        return self.bin_op(self.xAND,(TT_ENTAILS))




    def bin_op(self,func,ops):
        res = ParseResult()
        left= res.register(func())
        if res.error: return res

        while self.current_tok.type in ops:
            op_tok = self.current_tok
            res.register(self.advance())
            right = res.register(func())
            if res.error: return res
            left = Bin0pNode(left ,op_tok,right)

        return res.success(left)

    def un_op(self,func,ops):
        res = ParseResult()
        left= res.register(func())
        if res.error: return res

        while self.current_tok.type in ops:
            op_tok = self.current_tok
            res.register(self.advance())
            right = res.register(func())
            if res.error: return res
            left = UnaryOpNode(op_tok,right)

        return res.success(left)



###############################
#Values
###############################


class Laws:
    def __init__(self,expression,v1,v2):
        self.expression = expression
        self.v1 = v1
        self.v2 = v2

    def __repr__(self):
        return f'{self.expression},{self.v1},{self.v2}'

    def self_contradiction(self,expression,left,right):

        self.left = left
        self.right = right

        x = str(left)
        z = str(left)
        y =str(right)
        w = str(right)
        final = Laws(expression,z,w)
        x = x.replace(')','')[-1:]
        y = y.replace(')','')[-1:]
    #    print(x,y)
        if len(z) != len(w) and x == y :
            print('Self-contradiction')
            print(final)

    def middles(self,expression,left,right):
        self.left = left
        self.right = right

        x = str(left)
        z = str(left)
        y =str(right)
        w = str(right)
        x = x.replace(')','')[-1:]
        y = y.replace(')','')[-1:]
    #    print(x,y)
        final = Laws(expression,z,w)

        if len(z) != len(w) and x == y :
            print('excluding middle P or ~ P')
            print(final)


    def tautology(self,expression,left,right):
        self.left = left
        self.right = right
        x = str(left)
        z = str(left)
        y =str(right)
        w = str(right)
        x = x.replace(')','')[-1:]
        y = y.replace(')','')[-1:]
        final = Laws(expression,z,w)

    #    print(x,y)
        if len(z) == len(w) and x == y :
            print('identity')
            print(final)
            return w






class VAR:
    def __init__( self, data=None):
        self.data = data.replace(' ','')


    def __repr__(self):
        return f'{self.data}'


class CU:
    def __init__( self, data, next ):
        self.data = data
        self.next = next

    def __repr__(self):
        return f'{self.data}{self.next}'

class TR:
    def __init__( self, data=None):
        self.data = str(data).replace(' ','')

    def __repr__(self):
        return f'{self.data}'

class TR0:
    def __init__( self, data):
        self.data = data


    def lister(self,cind):
        self.cind = cind
        print(cind,'cind')
        if cind[0] == '[' and cind[-1:] == ']':
            result = cind[1:-1].split('][')
            result[0]= result[0].split(',')
            result[1]= result[1].split(',')

            for j in range(0,2):
                for i in range(0,len(result[j])):
                    result[j][i]= VAR(result[j][i])

            results = []
            for i in range(0,len(result[0])):
                for j in range(0,len(result[1])):
                    re = Conj.bas(self,result[0][i],result[1][j])
                    re = VAR(re)
                    results.append(re)
            return results

        elif cind[0] == '[' and cind[-1:] != ']':
            result = cind[1:-1].split(']')
            result[0]= result[0].split(',')
            r1= cind[-1:]
            for i in range(0,len(result[0])):
                result[0][i]= VAR(result[0][i])

            results = []
            for i in range(0,len(result[0])):

                re = Conj.bas(self,result[0][i],r1)
                re = VAR(re)
                results.append(re)

            return results

        elif cind[0] != '[' and cind[-1:] == ']':
            result = cind[:-1].split('[')
            print(result)
            r0 = result[0]
        #    print()
            result[1]= result[1].split(',')

            for i in range(0,len(result[1])):
                result[1][i]= VAR(result[1][i])

            results = []
            for i in range(0,len(result[1])):
                re = Conj.bas(self,r0,result[1][i])
                re = VAR(re)
                results.append(re)

            return results


        elif cind[0] != '[' and cind[-1:] != ']':
            return cind




class Conj:
    def __init__( self, data, next ):
        self.data = data
        self.next = next

    def bas(self,data,next):
        self.data = data
        self.next = next
        print('there',data,next)
        result = str(data) + str(next)
        return result


class Union:
    def __init__( self, data, next ):
        self.data = data
        self.next = next


    def super(self,thing,next):
        self.next = next
        self.thing = thing
        result = str(thing).replace('[','').replace(']','')#.replace(' ','')
        result = result.split(',')
    #    print(thing)

        for i in range(0,len(result)):
            if not isinstance(next,list):
                result[i] = TR(result[i])
                result.append(CU(result[i],next))
            elif isinstance(next,list) and not isinstance(thing,list) :
                for j in range(0,len(next)):
                    result[i] = TR(result[i])
                    result.append(CU(result[i],next[j]))

                    result.append(TR(next[j]))
                return result
            elif isinstance(next,list) and  isinstance(thing,list) :
                result = []
                for k in range(0,len(thing)):
                    thing[k] = TR(thing[k])
                    result.append(thing[k])
                    for j in range(0,len(next)):
                        result.append(CU(thing[k],next[j]))
                for j in range(0,len(next)):
                    result.append(TR(next[j]))
                return result

        result.append(TR(next))

        return result


class VAR2:
    def __init__( self, data=None):
        self.data = data

    def negation(self,tok,value):
        self.value = value
        self.tok = tok
    #    print(value,'here')
        if str(tok) == '~':
            if not isinstance(value,str) and not isinstance(value,list) :
                 value =  str(tok) + str(value)
            elif isinstance(value,list) :
                for i in range(0,len(value)):
                    if len(str(value[i])) == 1:
                        value[i] = str(tok) + str(value[i])
                        value[i] = VAR(value[i])
                    else:
                        value2 = str()
                        x = len(str(value[i]))
                        j = 0
                        while j < x:
                            if str(value[i])[j] != '~':
                                value2 += str(tok)+str(value[i])[j]
                            elif str(value[i])[j] == '~':
                                j += 1
                                value2 += str(value[i])[j]
                            j += 1
                        value[i] = value2
                        value[i] = VAR(value[i])
            #    for i in range(0,len(value)):
            #        if str(value[i])[0:2] == '~~':
            #            value[i] = str(value[i])[2:]
            #            value[i] = VAR(value[i])


            elif isinstance(value,str) :
                value2 = str()
                x = len(value)
                i = 0
                while i < x:
                    if value[i] != '~':
                        value2 += str(tok)+value[i]
                    elif value[i] == '~':
                        i += 1
                        value2 += value[i]
                    i += 1

                value = value2
        return value



###############################
#Context
###############################

class Context:
    def __init__(self,display_name, parent = None, parent_entry_pos=None):
        self.display_name = display_name
        self.parent= parent
        self.parent_entry_pos= parent_entry_pos



###############################
#Interpeter
###############################



class Interpeter:
    def visit(self,node):
        method_name=f'visit_{type(node).__name__}'
        method = getattr(self,method_name, self.no_visit_method)
    #    print('node', node,'here')
        return method(node)

    def goto(self,node):
        method_name=f'goto_{type(node).__name__}'
        method = getattr(self,method_name, self.no_goto_method)
#        print('not node', type(node),'here')
        return method(node)

    def no_visit_method(self,node):
        raise Exception(f'No visit_{type(node).__name__} method defined')

    def no_goto_method(self,node):
        raise Exception(f'No goto_{type(node).__name__} method defined')


    def visit_InputNode(self,node):
        self.node = node
        return VAR(node.tok.value)

    def goto_InputNode(self,node):
        self.node = node
        return VAR(node.tok.value)

    def visit_Bin0pNode(self,node):
        self.node = node
#        print('visit LR')
        left = self.visit(node.left_node)########
        right = self.visit(node.right_node)######



        if node.op_tok.value == '\\/':
            Laws.self_contradiction(self,'AND',node.left_node,node.right_node)
            Laws.tautology(self,'AND',node.left_node,node.right_node)
            result = Conj.bas(self,left,right)
    #        reverse = Union.super(self,left,right)
    #
        elif node.op_tok.value == '/\\':
            Laws.middles(self,'OR',node.left_node,node.right_node)
            Laws.tautology(self,'OR',node.left_node,node.right_node)
            result = Union.super(self,left,right)
            print(left,right)

    #        reverse = Conj.bas(self,left,right)

        if isinstance(result,str):
    #        print(type(left),right,'here')
            result = TR0.lister(self,result)


        return result

    def goto_Bin0pNode(self,node):
        self.node = node

        left = self.goto(node.left_node)#########
        right = self.goto(node.right_node)########
#        print('goto LR',type(left),type(right))

        if node.op_tok.value == '/\\':
        #    Laws.self_contradiction(self,'AND',node.left_node,node.right_node)
        #    Laws.tautology(self,'AND',node.left_node,node.right_node)
            reverse = Conj.bas(self,left,right)
#            print(left,right,'xOR')

        elif node.op_tok.value == '\\/':
        #    Laws.middles(self,'OR',node.left_node,node.right_node)
        #    Laws.tautology(self,'OR',node.left_node,node.right_node)

            reverse = Union.super(self,left,right)
            print(left,right,'xAND')

        if isinstance(reverse,str):
            reverse = TR0.lister(self,reverse)


        return reverse


    def visit_UnaryOpNode(self,node):
        self.node = node
        if node.op_tok.value == '~':
            result = VAR2.negation(self,node.op_tok.value,self.goto(node.right_node))
            #print(node,'mgoto')


        return result



    def goto_UnaryOpNode(self,node):
        self.node = node
        if node.op_tok.value == '~':
            result = VAR2.negation(self,node.op_tok.value,self.visit(node.right_node))
            #print(node,'mvisit')

        return result


###############################
###############################
#Run

def run(fn,text):
    #generate Tokens
    lexer = Lexer(fn,text)
    tokens, error = lexer.make_tokens()
    if error: return None, error

    #generate AST
    parser = Parser(tokens)
    ast = parser.parse()
    if ast.error: return None, ast.error

    #Run programm
    interpreter = Interpeter()
    context = Context('<program>')
    result = interpreter.visit(ast.node)

    return result, None #ast.node, ast.error
