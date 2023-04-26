#Segunda parte: en esta parte entrenaremos el modelo de NPL


import argparse
import os
import torch

from flair.datasets import ColumnCorpus
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', '-p', type=str, required=True, help='Path to data files')
parser.add_argument('--storage-path', type=str, default='taggers/', help='Basepath to store model weights and logs')
parser.add_argument('--tag_type', type=str, default='ner', help='Tag type to be used in flair models')

#Parametros del modelo


#Parametros de entrenamiento

args = parser.parse_args()

tag_type = args.tag_type

colums = {0: 'text', 1: "decoded_text", 2: 'begin', 3: 'end', 4: tag_type}
corpus = ColumnCorpus(

)