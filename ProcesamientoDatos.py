#Primera parte del código para desarrollar la tarea Meddoplace: https://temu.bsc.es/meddoplace/


import argparse
import os
import re

parser = argparse.ArgumentParser()
parser.add_argument('--train_files', type=str, default='D:\\uni\\TFG\\Datos\\meddoplace_train_set\\training_set\\brat\\meddoplace_brat_train', help='Ruta de los archivos de entrenamiento (.ann y .txt)')
parser.add_argument('--out_files', type=str, default='D:\\uni\\TFG\\Datos\\Datos_tokenizados', help='Ruta donde se guardan los archivos tokenizados')
args = parser.parse_args()

if __name__ == '__main__':

    print(args.train_files)
    print(args.out_files)
    files_path = args.train_files
    out_path = args.out_files