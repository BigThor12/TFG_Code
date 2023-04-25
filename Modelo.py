#Segunda parte: en esta parte entrenaremos el modelo de NPL


import argparse
import os
import torch

from flair.datasets import ColumnCorpus
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer

