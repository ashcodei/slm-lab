"""Read primitive-only legacy dataset metadata; never resolve or execute globals."""
import io
import pickle
import pickletools

ALLOWED={'PROTO','FRAME','STOP','MARK','NONE','NEWTRUE','NEWFALSE','INT','BININT','BININT1','BININT2',
         'LONG','LONG1','LONG4','FLOAT','BINFLOAT','STRING','BINSTRING','SHORT_BINSTRING',
         'UNICODE','BINUNICODE','SHORT_BINUNICODE','BINUNICODE8','BINBYTES','SHORT_BINBYTES','BINBYTES8',
         'EMPTY_DICT','DICT','SETITEM','SETITEMS','EMPTY_LIST','LIST','APPEND','APPENDS',
         'EMPTY_TUPLE','TUPLE','TUPLE1','TUPLE2','TUPLE3','EMPTY_SET','ADDITEMS','FROZENSET',
         'BINPUT','LONG_BINPUT','BINGET','LONG_BINGET','GET','PUT','MEMOIZE','POP','POP_MARK','DUP'}


class PrimitiveUnpickler(pickle.Unpickler):
    def find_class(self,module,name):raise ValueError('Dataset metadata cannot resolve Python globals')
    def persistent_load(self,pid):raise ValueError('Dataset metadata cannot load persistent objects')


def load_primitives(raw,max_bytes=300_000_000):
    if len(raw)>max_bytes:raise ValueError('Dataset metadata exceeds size bound')
    stopped=False
    for opcode,argument,position in pickletools.genops(raw):
        if opcode.name not in ALLOWED:raise ValueError('Executable/unsupported pickle opcode: '+opcode.name)
        if opcode.name in {'BINPUT','LONG_BINPUT','BINGET','LONG_BINGET','GET','PUT'} and int(argument)>2_000_000:
            raise ValueError('Dataset metadata memo index exceeds bound')
        if opcode.name=='STOP':
            if position+1!=len(raw):raise ValueError('Trailing metadata after pickle object')
            stopped=True
    if not stopped:raise ValueError('Unterminated metadata object')
    return PrimitiveUnpickler(io.BytesIO(raw)).load()
