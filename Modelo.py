#Segunda parte: en esta parte entrenaremos el modelo de NPL


import argparse
import os
import torch

from flair.datasets import ColumnCorpus
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer

parser = argparse.ArgumentParser()
parser.add_argument('--train_path', '-p', type=str, default='Datos\\Training_files\\', help='Path to data files')
parser.add_argument('--storage-path', type=str, default='taggers/', help='Basepath to store model weights and logs')
parser.add_argument('--tag_type', type=str, default='ner', help='Tag type to be used in flair models')

#Parametros del modelo


#Parametros de entrenamiento

args = parser.parse_args()

tag_type = args.tag_type

colums = {0: 'text', 1: "decoded_text", 2: 'begin', 3: 'end', 4: tag_type}

corpus = ColumnCorpus(args.train_path, colums,
                      train_file= 'train_ready.txt',
                      test_file= 'train_test_ready.txt',
                      dev_file= 'train_test_ready.txt',
                      tag_to_bioes=tag_type,
                      document_separator_token='<DOCSTART>'
)