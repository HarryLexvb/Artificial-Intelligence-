import argparse #esta biblioteca es para poder leer los argumentos que se le pasan al programa
import os #esta biblioteca es para poder leer los argumentos que se le pasan al programa
from pathlib import Path #esta biblioteca es para poder leer los argumentos que se le pasan al programa

from synthesizer.hparams import hparams #esta biblioteca es para poder leer los argumentos que se le pasan al programa
from synthesizer.synthesize import run_synthesis #esta biblioteca es para poder leer los argumentos que se le pasan al programa
from utils.argutils import print_args #esta biblioteca es para poder leer los argumentos que se le pasan al programa

if __name__ == "__main__":
    class MyFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter):
        pass

    parser = argparse.ArgumentParser(
        description="Creates ground-truth aligned (GTA) spectrograms from the vocoder.",
        formatter_class=MyFormatter
    )
    parser.add_argument("datasets_root", type=Path, help=\
        "Path to the directory containing your SV2TTS directory. If you specify both --in_dir and "
        "--out_dir, this argument won't be used.")
    parser.add_argument("-s", "--syn_model_fpath", type=Path,
                        default="saved_models/default/synthesizer.pt",
                        help="Path to a saved synthesizer") #con -s se le pasa el modelo que se quiere usar
    parser.add_argument("-i", "--in_dir", type=Path, default=argparse.SUPPRESS, help= \
        "Path to the synthesizer directory that contains the mel spectrograms, the wavs and the "
        "embeds. Defaults to  <datasets_root>/SV2TTS/synthesizer/.") #con -i se le pasa el directorio donde se encuentran los archivos de entrada
    parser.add_argument("-o", "--out_dir", type=Path, default=argparse.SUPPRESS, help= \
        "Path to the output vocoder directory that will contain the ground truth aligned mel "
        "spectrograms. Defaults to <datasets_root>/SV2TTS/vocoder/.") #con -o se le pasa el directorio donde se encuentran los archivos de salida
    parser.add_argument("--hparams", default="", help=\
        "Hyperparameter overrides as a comma-separated list of name=value pairs") #con --hparams se le pasa los hiperparametros que se quieren usar
    parser.add_argument("--cpu", action="store_true", help=\
        "If True, processing is done on CPU, even when a GPU is available.") #con --cpu se le pasa si se quiere usar la cpu o no
    args = parser.parse_args() #lee los argumentos que se le pasan al programa
    print_args(args, parser) #imprime los argumentos que se le pasan al programa
    modified_hp = hparams.parse(args.hparams) #lee los argumentos que se le pasan al programa

    if not hasattr(args, "in_dir"):
        args.in_dir = args.datasets_root / "SV2TTS" / "synthesizer"
    if not hasattr(args, "out_dir"):
        args.out_dir = args.datasets_root / "SV2TTS" / "vocoder"

    if args.cpu:
        # Hide GPUs from Pytorch to force CPU processing
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    run_synthesis(args.in_dir, args.out_dir, args.syn_model_fpath, modified_hp)
