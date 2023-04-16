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


def get_offsets(text,tokens,offset):
    #Consigue el offset del texto para una lista de tokens
    found =  []
    try:
        for token in tokens:



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
        offsets =

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