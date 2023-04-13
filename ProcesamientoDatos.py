#Primera parte del código para desarrollar la tarea Meddoplace: https://temu.bsc.es/meddoplace/


import argparse
import os
import re

parser = argparse.ArgumentParser()
parser.add_argument('--train_files', type=str, default='Datos\\meddoplace_train_set\\training_set\\brat\\meddoplace_brat_train\\', help='Ruta de los archivos de entrenamiento (.ann y .txt)')
parser.add_argument('--out_files', type=str, default='Datos\\Datos_tokenizados\\', help='Ruta donde se guardan los archivos tokenizados')
args = parser.parse_args()

def leer_archivo(txt_file):
    #Función para leer el contenido de un archivo txt
    with open(txt_file, 'r', encoding='utf-8') as txt:
        text = txt.read()

    return text

def tokenizacion_del_archivo(ann_file,text_file,file_key = '-'):

    return


def procesamiento_de_ficheros(files_path,out_path):
    #Recorremos todos los fichero ann y txt y los combinamos y tokenizamos. Por último creamos el archivo tokenizado y lo guardamos en out_path para su posterior uso

    ann_files = sorted([files_path + x for x in os.listdir(files_path) if x.endswith('.ann')])
    txt_files = sorted([files_path + x for x in os.listdir(files_path) if x.endswith('.txt')])

    for ann_file, txt_file in zip(ann_files,txt_files):
        file_key = txt_file.split("\\")[-1][:-4]
        #print(file_key)
        conll_format_file = tokenizacion_del_archivo(ann_file,txt_file,file_key)


if __name__ == '__main__':

    #print(args.train_files)
    #print(args.out_files)
    procesamiento_de_ficheros(args.train_files,args.out_files)