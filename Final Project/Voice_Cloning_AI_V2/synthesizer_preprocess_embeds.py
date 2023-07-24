from synthesizer.preprocess import create_embeddings #esta biblioteca es la que se encarga de crear los embeddings
from utils.argutils import print_args #esta biblioteca es la que se encarga de imprimir los argumentos
from pathlib import Path #esta biblioteca es la que se encarga de manejar los paths
import argparse #esta biblioteca es la que se encarga de manejar los argumentos


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Crea incrustaciones para el sintetizador a partir de los enunciados de LibriSpeech.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    ) #crea un objeto de la clase ArgumentParser
    parser.add_argument("synthesizer_root", type=Path, help=\
        "Ruta a los datos de entrenamiento del sintetizador que contiene los audios y el archivo train.txt. "
        "Si dejas todo por defecto, debería ser <datasets_root>/SV2TTS/synthesizer/.")
    parser.add_argument("-e", "--encoder_model_fpath", type=Path,
                        default="saved_models/default/encoder.pt", help=\
        "Path your trained encoder model.") #establece el path del encoder
    parser.add_argument("-n", "--n_processes", type=int, default=4, help= \
        "Número de procesos paralelos. Se crea un codificador para cada uno, por lo que puede ser necesario reducir "
        "este valor en GPUs con poca memoria. Establézcalo a 1 si CUDA no está satisfecho.")
    args = parser.parse_args() #parsea los argumentos

    # Preprocess the dataset
    print_args(args, parser) #imprime los argumentos
    create_embeddings(**vars(args))
