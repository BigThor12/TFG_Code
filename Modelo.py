#Segunda parte: en esta parte entrenaremos el modelo de NPL


import argparse
import os
import torch

from flair.datasets import ColumnCorpus
from flair.embeddings import TransformerWordEmbeddings
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer

parser = argparse.ArgumentParser()
parser.add_argument('--train_path', '-p', type=str, default='Datos\\Training_files\\', help='Path to data files')
parser.add_argument('--storage-path', type=str, default='taggers\\', help='Basepath to store model weights and logs')
parser.add_argument('--label_type', type=str, default='ner', help='Tag type to be used in flair models')



args = parser.parse_args()

label_type = args.label_type

colums = {0: 'text', 1: "decoded_text", 2: 'begin', 3: 'end', 4: label_type}

corpus = ColumnCorpus(args.train_path, colums,
                      train_file= 'train_ready.txt',
                      test_file= 'train_test_ready.txt',
                      dev_file= 'train_test_ready.txt',
                      tag_to_bioes=label_type,
                      document_separator_token='<DOCSTART>',
                      encoding="utf-16"
                      )

label_dict = corpus.make_label_dictionary(label_type=label_type)
print(label_dict)

embeddings = TransformerWordEmbeddings(model= 'xlm-roberta-large',
                                       layers="-1",
                                       subtoken_pooling="first",
                                       fine_tune=False,
                                       use_context=True)

tagger = SequenceTagger(hidden_size=256,
                        embeddings=embeddings,
                        tag_dictionary=label_dict,
                        tag_type='ner',
                        use_crf=True,
                        use_rnn=True,
                        reproject_embeddings=False)

trainer = ModelTrainer(tagger, corpus)

trainer.train(args.storage_path+"test",
              learning_rate=0.1,
              mini_batch_size=32,
              max_epochs=150)

