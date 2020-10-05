import codecs
import pdb

def spm(input_file, output_file, spm_model=None):
    assert spm_model
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor()
    sp.Load(spm_model)
    with codecs.open(input_file, encoding='utf-8') as f:
        lines = f.readlines()
        
    lines = [sp.DecodePieces(x.rstrip().split()) for x in lines]
    with codecs.open(output_file, "w", encoding='utf-8') as f:
        f.write("\n".join(lines))

def get_detok_func(fn_name):

    if fn_name=='spm':
        return eval(fn_name)
    else:
        raise ValueError(f'{fn_name} is not a valid detokenization function') 


#input_file = '../newstest2014-src.spm.en.output'
#output_file = '../newstest2014-src.spm.en.output.test'
#spm_model = '../enfr_vocab.model'
#spm(input_file, output_file, spm_model)
