# TFG_Code
En este repositorio se encuentra el código que he programado para realizar mi TFG. Este se centra en la tarea MeddoPlace, descrita en el siguiente enlace:  https://temu.bsc.es/meddoplace/

### Creación del entorno
El código se ejecuta dentro de un entorno _conda_. Este se puede crear usando los siguientes comandos:

    conda create -n meddoplace python==3.8.5
    conda activate meddoplace
    pip install flair==0.8 transformers==4.6.1 torch==1.8.1 scikit-learn==0.23.1 scipy==1.6.3 numpy==1.19.5 nltk tqdm seaborn matplotlib