#Primera parte del código para desarrollar la tarea Meddoplace: https://temu.bsc.es/meddoplace/
#Este codigo esta hecho usando de referencia el siguiente repositorio de github: https://github.com/boschresearch/nlnde-meddoprof


import argparse
import os
import re
from transformers import AutoTokenizer
from tqdm import tqdm

from sentencesplit import sentencebreaks_to_newlines


from collections import namedtuple
Token = namedtuple('Token','encID text lema pos file_key sent_id token_id start end labels')
Annotation = namedtuple('Annotation',['tid','type','start','end','text'])

parser = argparse.ArgumentParser()
parser.add_argument('--train_files', type=str, default='Datos\\meddoplace_train_set\\training_set\\brat\\meddoplace_brat_train\\', help='Ruta de los archivos de entrenamiento (.ann y .txt)')
parser.add_argument('--out_files', type=str, default='Datos\\Datos_tokenizados\\', help='Ruta donde se guardan los archivos tokenizados')
parser.add_argument('--model', '-m', type=str, default='xlm-roberta-large', help='Path to/Name of huggingface model')
parser.add_argument('--max_anidacion', type=int, default=1, help='Añade niveles de anidación')
args = parser.parse_args()

NEWLINE_TERM_REGEX = re.compile(r'(.*?\n)')
#Grupo 1: id de la anotacion; Grupo 2: tipo; Grupo 3: start; Grupo 4: end; Grupo 5: texto al que hace referencia
T_ANNOTATION_REGEX = re.compile(r'(T\d+)\s+(\S*?)\s(\d+)\s(\d+)\s(.*)', re.MULTILINE)
#Grupo 1: id (Ax); Grupo 2: tipo; Grupo 3: id de la anotacion a la que hace referencia
A_ANNOTATION_REGEX = re.compile(r'(A\d+)\s+(\S*?)\s(T\d+)', re.MULTILINE)
#Grupo 1: id (#x); Grupo 2: "AnnotatorNotes"; Grupo 3: id de la anotacion a la que hace referencia; Grupo 4: tipo; Grupo 5: id numerico
H_ANNOTATION_REGEX = re.compile(r'(#\d+)\s(\S*?)\s(T\d+)\s(\S*?):(\d+)', re.MULTILINE)

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
            if token[0] == '▁':
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
        sentences.extend([s for s in NEWLINE_TERM_REGEX.split(t) if s])

    content = [[Token('<DOCSTART>', '<DOCSTART>', '-x-', '-y-', file_key, -1, -1, 0, 0, [])]]
    last_offset = 0
    for s_id, s in enumerate(sentences):
        enc = tokenizer.encode(s)
        org_tokens = tokenizer.convert_ids_to_tokens(enc)
        if len(org_tokens) == 0:
            continue
        tokens = org_tokens[1:-1] #Quitar los tokens especiales
        if len(tokens) == 0:
            continue
        offsets = get_offsets(text_content, tokens, last_offset)

        all_tokens = [(str(enc[t_id+1]), t, '-x-', '-y-', file_key, s_id, t_id, offsets[t_id][0], offsets[t_id][1]) for t_id,t in enumerate(tokens)]
        tokens = [Token(*s, []) for s in all_tokens]
        content.append(tokens)
        last_offset = tokens[-1].end

    return content


def leer_anotaciones(ann_content):
    #Lee las anotacions de un documento de anotaciones
    annotations = []
    for a in T_ANNOTATION_REGEX.finditer(ann_content):
        ann_id = a.group(1)
        ann_type = a.group(2)
        ann_begin = int(a.group(3))
        ann_end = int(a.group(4))
        ann_text = a.group(5)
        annotations.append(Annotation(ann_id,ann_type,ann_begin,ann_end,ann_text))

    return annotations


def separar_anotaciones_anidadas(annotations,max_level=0):

    nested_annotations = {}
    nested_level = 0

    while len(annotations) > 0 and nested_level < max_level:
        nested_annotations[nested_level] = []
        eliminate = {}

        for a1 in annotations:
            for a2 in annotations:
                if a1 is a2: #si son el mismo
                    continue
                if a2.start >= a1.end or a2.end <= a1.start: #si no se superponen
                    continue
                #Eliminamos la mas corta de las dos, ya que ya sabemos que estan anidadas
                if a1.end - a1.start > a2.end - a1.start:
                    eliminate[a2] = True
                else:
                    eliminate[a1] = True

        nested_annotations[nested_level] = [a for a in annotations if a not in eliminate]
        annotations = [a for a in annotations if a in eliminate]
        nested_level += 1

    nested_level = len(nested_annotations)
    if nested_level < max_level:
        for n in range(nested_level,max_level+1):
            nested_annotations[n] = [n]
    return nested_annotations,nested_level


def anadir_anotaciones_a_texto(document,annotations):
    #En esta función juntamos las anotaciones con el texto, añadiendo la nomenclatura BIO
    labels = ['O' for _ in range(document[-1][-1].end)]
    for a in annotations:
        labels[a.start] = 'B-' + a.type
        for y in (a.start+1,a.end):
            labels[y] = 'I-' + a.type

    prev_ann = 't-1'
    for sid, sentence in enumerate(document):
        if sid == 0 #Comienzo del documento
            for token in sentence:
                assert token.text == '<DOCSTART>'
                token.labels.append('O')

        else:
            for token in sentence:
                token.labels.append(labels[token.start])





def tokenizacion_del_archivo(ann_file,text_file,tokenizer,file_key = '-'):
    #Esta función sirve para juntar todos los pasos en la tokenización del archivo

    text_content = leer_archivo(text_file)
    ann_content = leer_archivo(ann_file)

    new_content = transformar_texto_a_conll(tokenizer,text_content,file_key)
    annotations = leer_anotaciones(ann_content)
    nested_annotations, nested_level = separar_anotaciones_anidadas(annotations,args.max_anidacion)
    for n in range(args.max_anidación):
        anadir_anotaciones_a_texto(new_content,nested_annotations[n])
    conll_file =
    return


def procesamiento_de_ficheros(files_path,out_path,tokenizer):
    #Recorremos todos los fichero ann y txt y los combinamos y tokenizamos. Por último creamos el archivo tokenizado y lo guardamos en out_path para su posterior uso

    ann_files = sorted([files_path + x for x in os.listdir(files_path) if x.endswith('.ann')])
    txt_files = sorted([files_path + x for x in os.listdir(files_path) if x.endswith('.txt')])

    t = tqdm(total=len(ann_files))
    num_proc = 0
    for ann_file, txt_file in zip(ann_files,txt_files):
        file_key = txt_file.split("\\")[-1][:-4]
        #print(file_key)
        conll_format_file = tokenizacion_del_archivo(ann_file,txt_file,tokenizer,file_key)

        with open(out_path + file_key + '.bio', 'w') as f:
            f.write(conll_format_file)

        num_proc += 1
        t.update()



if __name__ == '__main__':

    hf_tokenizer = AutoTokenizer.from_pretrained(args.model)

    #print(args.train_files)
    #print(args.out_files)
    procesamiento_de_ficheros(args.train_files,args.out_files,hf_tokenizer)
    #test_files