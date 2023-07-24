from synthesizer.preprocess import preprocess_dataset # esta biblioteca es para preprocesar los datos
from synthesizer.hparams import hparams # esta biblioteca es para los hiperparametros
from utils.argutils import print_args # esta biblioteca es para imprimir los argumentos
from pathlib import Path # esta biblioteca es para manejar los directorios
import argparse # esta biblioteca es para manejar los argumentos


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocesa archivos de audio a partir de conjuntos de datos, los codifica como espectrogramas mel "
                    "y los escribe en el disco. Los archivos de audio también se guardan, para ser utilizados por el "
                    "vocoder para el entrenamiento.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter # esta clase es para dar formato a los argumentos
    )
    parser.add_argument("datasets_root", type=Path, help=\
        "Path to the directory containing your LibriSpeech/TTS datasets.") # este argumento es para el directorio que contiene los datos
    parser.add_argument("-o", "--out_dir", type=Path, default=argparse.SUPPRESS, help=\
        "Path to the output directory that will contain the mel spectrograms, the audios and the "
        "embeds. Defaults to <datasets_root>/SV2TTS/synthesizer/") # este argumento es para el directorio de salida
    parser.add_argument("-n", "--n_processes", type=int, default=4, help=\
        "Number of processes in parallel.") # este argumento es para el numero de procesos en paralelo
    parser.add_argument("-s", "--skip_existing", action="store_true", help=\
        "Whether to overwrite existing files with the same name. Useful if the preprocessing was "
        "interrupted.") # este argumento es para sobreescribir archivos existentes con el mismo nombre
    parser.add_argument("--hparams", type=str, default="", help=\
        "Hyperparameter overrides as a comma-separated list of name-value pairs")
    parser.add_argument("--no_alignments", action="store_true", help=\
        "Use this option when dataset does not include alignments\
        (these are used to split long audio files into sub-utterances.)") # este argumento es para no usar alineaciones
    parser.add_argument("--datasets_name", type=str, default="LibriSpeech", help=\
        "Name of the dataset directory to process.") # este argumento es para el nombre del directorio de datos
    parser.add_argument("--subfolders", type=str, default="train-clean-100,train-clean-360", help=\
        "Comma-separated list of subfolders to process inside your dataset directory") # este argumento es para la lista de subdirectorios a procesar dentro del directorio de datos
    args = parser.parse_args()

    # Process the arguments
    if not hasattr(args, "out_dir"): # si no tiene el atributo out_dir
        args.out_dir = args.datasets_root.joinpath("SV2TTS", "synthesizer") # se crea el directorio de salida

    # Create directories
    assert args.datasets_root.exists() # se asegura que el directorio de datos exista
    args.out_dir.mkdir(exist_ok=True, parents=True) # se crea el directorio de salida

    # Preprocess the dataset
    print_args(args, parser) # se imprimen los argumentos
    args.hparams = hparams.parse(args.hparams) # se parsean los hiperparametros
    preprocess_dataset(**vars(args)) # se preprocesan los datos
