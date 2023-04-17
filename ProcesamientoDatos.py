#Primera parte del código para desarrollar la tarea Meddoplace: https://temu.bsc.es/meddoplace/
#Este codigo esta hecho usando de referencia el siguiente repositorio de github: https://github.com/boschresearch/nlnde-meddoprof


import argparse
import os
import re
from transformers import AutoTokenizer

from sentencesplit import sentencebreaks_to_newlines
NEWLINE_TERM_REGEX = re.compile(r'(.*?\n)')

from collections import namedtuple
Token = namedtuple('Token','encID text lema pos file_key sent_id token_id start end labels')
Annotation = namedtuple('Annotation'['tid','type','start','end','text'])

parser = argparse.ArgumentParser()
parser.add_argument('--train_files', type=str, default='Datos\\meddoplace_train_set\\training_set\\brat\\meddoplace_brat_train\\', help='Ruta de los archivos de entrenamiento (.ann y .txt)')
parser.add_argument('--out_files', type=str, default='Datos\\Datos_tokenizados\\', help='Ruta donde se guardan los archivos tokenizados')
args = parser.parse_args()

def leer_archivo(txt_file):
    #Función para leer el contenido de un archivo txt
    with open(txt_file, 'r', encoding='utf-8') as txt:
        text = txt.read()

    return text


def get_indice_del_token(text,token,offset):
    #Devuelve el el offset del texto para un token
    l = len(token)

    for s in range(offset, len(text)):
        if text[s:s + l] == token:
            return s, s + l

        if s > offset + 100:
            raise ValueError('Takes too long')

        #
        # There are some special tokens converted by XLM-R that have to be reconverted
        #

        # simple replacements
        for org, rep, off in [('º', 'o', 0), ('ª', 'a', 0), ('´', '_', 0),
                              ('µ', 'μ', 0), ('ü', 'ü', 0), ('ñ', 'ñ', 0),
                              ('²', '2', 0), ('³', '3', 0), ('É', 'É', 0),
                              ('ö', 'ö', 0), ('Í', 'Í', 0), ('Ó', 'Ó', 0),
                              ('Ú', 'Ú', 0), ('´', '▁́', -1)]:
            if rep in token and org in text[s:s + l]:
                return s, s + l + off

        # fixes problems where XLM-R replacement is longer/shorter than real text
        if token == '1⁄2' and text[s:s + 1] == '½':
            return s, s + 1
        if token == '1⁄4' and text[s:s + 1] == '¼':
            return s, s + 1
        if token == '3⁄4' and text[s:s + 1] == '¾':
            return s, s + 1
        if token == '...' and text[s:s + 1] == '…':
            return s, s + 1
        if token == '...)' and text[s:s + 2] == '…)':
            return s, s + 2
        if token == '...).' and text[s:s + 3] == '…).':
            return s, s + 3
        if token == '"...' and text[s:s + 2] == '"…':
            return s, s + 2
        if token == '<unk>':
            return s, s + 1
        if token == 'Á' and text[s:s + 2] == 'Á':
            return s, s + 2
        if token == 'ñ' and text[s:s + 2] == 'ñ':
            return s, s + 2
        if token == 'É' and text[s:s + 2] == 'É':
            return s, s + 2
        if token == 'RÁ' and text[s:s + 3] == 'RÁ':
            return s, s + 3
        if token == 'Í' and text[s:s + 2] == 'Í':
            return s, s + 2
        if token == 'Ó' and text[s:s + 2] == 'Ó':
            return s, s + 2
        if token == 'è' and text[s:s + 2] == 'è':
            return s, s + 2
        if token == 'Ú' and text[s:s + 2] == 'Ú':
            return s, s + 2
        if '™' in text[s - 1:] and token == text.replace('™', 'TM')[s:s + l]:
            return s, s + l - 1
        if 'ŀ' in text[s - 1:] and token == text.replace('ŀ', 'l·')[s:s + l]:
            return s, s + l - 1

    raise ValueError('No se ha podido machear el token: ' + token)


def get_offsets(text,tokens,offset):
    #Consigue el offset del texto para una lista de tokens
    found =  []
    try:
        for token in tokens:
            if token[0] == '_':
                try:
                    s1, e1 = get_indice_del_token(text,token,offset)
                except:
                    s1, e1 = len(text)+2, len(text)+3

                try:
                    s2, e2 = get_indice_del_token(text,token[1:],offset)
                except:
                    s2, e2 = len(text) + 2, len(text) + 3

                if s1 <= s2:
                    assert s1 <= len(text)
                    found.append((s1,e1))
                    offset = e1
                else:
                    assert s2 <= len(text)
                    found.append((s2,e2))
                    offset = e2

            else:
                s,e = get_indice_del_token(text,token,offset)
                assert s <= len(text)
                found.append((s,e))
                offset = e

    except:
        raise ValueError("Problema de Tokenización")
    return found


def transformar_texto_a_conll(tokenizer,text_content,file_key='-'):
    #En esta función separamos el texto en sentencias y tokens para dejarlo en formato conll
    sentences = []
    for t in text_content.splitlines():
        t = sentencebreaks_to_newlines(t)
        sentences.extend([s for s in NEWLINE_TERM_REGEX.split(1) if s])

    content = [[Token('<DOCSTART>', '<DOCSTART>', '-x-', '-y-', file_key, -1, -1, 0, 0, [])]]
    last_offset = 0
    for t_id, t in enumerate(sentences):
        enc = tokenizer.encode(t)
        org_tokens = tokenizer.convert_ids_to_tokens(enc)
        if len(org_tokens) == 0:
            continue
        tokens = org_tokens[1:-1] #Quitar los tokens especiales
        if len(tokens) == 0:
            continue
        offsets = get_offsets(text_content, tokens, last_offset)

def tokenizacion_del_archivo(ann_file,text_file,tokenizer,file_key = '-'):
    #Esta función sirve para juntar todos los pasos en la tokenización del archivo

    text_content = leer_archivo(text_file)
    ann_content = leer_archivo(ann_file)

    new_content = transformar_texto_a_conll(tokenizer,text_content,file_key)

    return


def procesamiento_de_ficheros(files_path,out_path,tokenizer):
    #Recorremos todos los fichero ann y txt y los combinamos y tokenizamos. Por último creamos el archivo tokenizado y lo guardamos en out_path para su posterior uso

    ann_files = sorted([files_path + x for x in os.listdir(files_path) if x.endswith('.ann')])
    txt_files = sorted([files_path + x for x in os.listdir(files_path) if x.endswith('.txt')])

    for ann_file, txt_file in zip(ann_files,txt_files):
        file_key = txt_file.split("\\")[-1][:-4]
        #print(file_key)
        conll_format_file = tokenizacion_del_archivo(ann_file,txt_file,file_key,tokenizer)


if __name__ == '__main__':

    hf_tokenizer = AutoTokenizer.from_pretrained()

    #print(args.train_files)
    #print(args.out_files)
    procesamiento_de_ficheros(args.train_files,args.out_files,hf_tokenizer)