import argparse
import os
from pathlib import Path
import librosa
import numpy as np
import soundfile as sf
import torch
import ast
import pprint
import inflect
import re
from unidecode import unidecode
from typing import Union, List

import urllib.request
from threading import Thread
from urllib.error import HTTPError
from tqdm import tqdm

speaker_embedding_size=256

from encoder1 import inference as encoder
from vocoder1 import inference as vocoder

from synthesizer1 import audio, cleaners
from synthesizer1.models.tacotron import Tacotron

#default_models
default_models = {
    "encoder": ("https://drive.google.com/uc?export=download&id=1q8mEGwCkFy23KZsinbuvdKAQLqNKbYf1", 17090379),
    "synthesizer": ("https://drive.google.com/u/0/uc?id=1EqFMIbvxffxtjiVrtykroF6_mUh-5Z3s&export=download&confirm=t", 370554559),
    "vocoder": ("https://drive.google.com/uc?export=download&id=1cf2NO6FtI0jDuy8AV3Xgn6leO6dHjIgu", 53845290),
}
class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)
def download(url: str, target: Path, bar_pos=0):
    target.parent.mkdir(exist_ok=True, parents=True)

    desc = f"Downloading {target.name}"
    with DownloadProgressBar(unit="B", unit_scale=True, miniters=1, desc=desc, position=bar_pos, leave=False) as t:
        try:
            urllib.request.urlretrieve(url, filename=target, reporthook=t.update_to)
        except HTTPError:
            return
def ensure_default_models(models_dir: Path):
    jobs = []
    for model_name, (url, size) in default_models.items():
        target_path = models_dir / "default" / f"{model_name}.pt"
        if target_path.exists():
            if target_path.stat().st_size != size:
                print(f"Archivo {target_path} no es del tamaño esperado...")
            else:
                continue

        thread = Thread(target=download, args=(url, target_path, len(jobs)))
        thread.start()
        jobs.append((thread, target_path, size))

    for thread, target_path, size in jobs:
        thread.join()

        assert target_path.exists() and target_path.stat().st_size == size, \
            f"Download for {target_path.name} failed. You may download models manually instead.\n" \
            f"https://drive.google.com/drive/folders/1fU6umc5uQAVR2udZdHX-lDgXYzTyqG_j"


_type_priorities = [
    Path,
    str,
    int,
    float,
    bool,
]
def _priority(o):
    p = next((i for i, t in enumerate(_type_priorities) if type(o) is t), None)
    if p is not None:
        return p
    p = next((i for i, t in enumerate(_type_priorities) if isinstance(o, t)), None)
    if p is not None:
        return p
    return len(_type_priorities)
def print_args(args: argparse.Namespace, parser=None):
    args = vars(args)
    if parser is None:
        priorities = list(map(_priority, args.values()))
    else:
        all_params = [a.dest for g in parser._action_groups for a in g._group_actions]
        priority = lambda p: all_params.index(p) if p in all_params else len(all_params)
        priorities = list(map(priority, args.keys()))

    pad = max(map(len, args.keys())) + 3
    indices = np.lexsort((list(args.keys()), priorities))
    items = list(args.items())

_pad        = "_"
_eos        = "~"
_characters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz!\'\"(),-.:;? "
symbols = [_pad, _eos] + list(_characters)

class HParams(object):
    def __init__(self, **kwargs): self.__dict__.update(kwargs)
    def __setitem__(self, key, value): setattr(self, key, value)
    def __getitem__(self, key): return getattr(self, key)
    def __repr__(self): return pprint.pformat(self.__dict__)

    def parse(self, string): # esta funcion es para que se pueda pasar los parametros por consola
        if len(string) > 0:
            overrides = [s.split("=") for s in string.split(",")]
            keys, values = zip(*overrides)
            keys = list(map(str.strip, keys))
            values = list(map(str.strip, values))
            for k in keys:
                self.__dict__[k] = ast.literal_eval(values[keys.index(k)])
        return self

hparams = HParams( # parametros de la red

        sample_rate = 16000, # frecuencia de muestreo
        n_fft = 800, # numero de muestras de la ventana
        num_mels = 80, # numero de filtros mel
        hop_size = 200, # numero de muestras que se desplaza la ventana
        win_size = 800, # numero de muestras de la ventana
        fmin = 55, # frecuencia minima
        min_level_db = -100, # nivel minimo de decibeles
        ref_level_db = 20, # nivel de referencia de decibeles
        max_abs_value = 4., # valor maximo absoluto
        preemphasis = 0.97, # preenfasis
        preemphasize = True, # preenfasis

        ### Tacotron Text-to-Speech (TTS)
        tts_embed_dims = 512,  # dimensiones de los embeddings
        tts_encoder_dims = 256, # dimensiones del encoder
        tts_decoder_dims = 128, # dimensiones del decoder
        tts_postnet_dims = 512, # dimensiones del postnet
        tts_encoder_K = 5,  # kernel size
        tts_lstm_dims = 1024, # dimensiones de la lstm
        tts_postnet_K = 5, # kernel size
        tts_num_highways = 4, # numero de capas de las carreteras
        tts_dropout = 0.5, # dropout
        tts_cleaner_names = ["english_cleaners"], # limpiador de texto
        tts_stop_threshold = -3.4, # umbral de parada para el decoder


        # entrenamiento de tacotron
        tts_schedule = [(2,  1e-3,  20_000,  12),
                        (2,  5e-4,  40_000,  12),
                        (2,  2e-4,  80_000,  12),
                        (2,  1e-4, 160_000,  12),
                        (2,  3e-5, 320_000,  12),
                        (2,  1e-5, 640_000,  12)],

        tts_clip_grad_norm = 1.0,
        tts_eval_interval = 500,                    # numero de iteraciones para evaluar


        tts_eval_num_samples = 1,

        #procesamiento de los datos
        max_mel_frames = 900,
        rescale = True,
        rescaling_max = 0.9,
        synthesis_batch_size = 16,

        #Mel Visualization and Griffin-Lim
        signal_normalization = True,
        power = 1.5,
        griffin_lim_iters = 60,

        # Audio processing options
        fmax = 7600,                                # maxima frecuencia para el mel-spectrograma
        allow_clipping_in_normalization = True,
        clip_mels_length = True,
        use_lws = False,
        symmetric_mels = True,

        trim_silence = True,

        ### SV2TTS
        speaker_embedding_size = 256,
        silence_min_duration_split = 0.4,
        utterance_min_duration = 1.6,
        )

def hparams_debug_string(): # esta funcion es para que se pueda pasar los parametros por consola
    return str(hparams)

##Cleaners

_whitespace_re = re.compile(r"\s+")
_abbreviations = [(re.compile("\\b%s\\." % x[0], re.IGNORECASE), x[1]) for x in [
    ("mrs", "misess"),
    ("mr", "mister"),
    ("dr", "doctor"),
    ("st", "saint"),
    ("co", "company"),
    ("jr", "junior"),
    ("maj", "major"),
    ("gen", "general"),
    ("drs", "doctors"),
    ("rev", "reverend"),
    ("lt", "lieutenant"),
    ("hon", "honorable"),
    ("sgt", "sergeant"),
    ("capt", "captain"),
    ("esq", "esquire"),
    ("ltd", "limited"),
    ("col", "colonel"),
    ("ft", "fort"),
]]

def expand_abbreviations(text):
    for regex, replacement in _abbreviations:
        text = re.sub(regex, replacement, text)
    return text
def expand_numbers(text):
    return normalize_numbers(text)
def lowercase(text):

    return text.lower()
def collapse_whitespace(text):
    return re.sub(_whitespace_re, " ", text)
def convert_to_ascii(text):
    return unidecode(text)
def basic_cleaners(text):
    #pipeline basico de limpieza de texto
    text = lowercase(text)
    text = collapse_whitespace(text)
    return text
def transliteration_cleaners(text):
    #pipline para transliterar texto a caracteres ASCII
    text = convert_to_ascii(text)
    text = lowercase(text)
    text = collapse_whitespace(text)
    return text
def english_cleaners(text):
    #pipeline para limpiar texto en ingles
    text = convert_to_ascii(text)
    text = lowercase(text)
    text = expand_numbers(text)
    text = expand_abbreviations(text)
    text = collapse_whitespace(text)
    return text

#normalize numbers

_inflect = inflect.engine()
_comma_number_re = re.compile(r"([0-9][0-9\,]+[0-9])")
_decimal_number_re = re.compile(r"([0-9]+\.[0-9]+)")
_pounds_re = re.compile(r"£([0-9\,]*[0-9]+)")
_dollars_re = re.compile(r"\$([0-9\.\,]*[0-9]+)")
_ordinal_re = re.compile(r"[0-9]+(st|nd|rd|th)")
_number_re = re.compile(r"[0-9]+")

def _remove_commas(m):
    return m.group(1).replace(",", "")
def _expand_decimal_point(m):
    return m.group(1).replace(".", " point ")
def _expand_dollars(m):
    match = m.group(1)
    parts = match.split(".")
    if len(parts) > 2:
        return match + " dollars"  # Unexpected format
    dollars = int(parts[0]) if parts[0] else 0
    cents = int(parts[1]) if len(parts) > 1 and parts[1] else 0
    if dollars and cents:
        dollar_unit = "dollar" if dollars == 1 else "dollars"
        cent_unit = "cent" if cents == 1 else "cents"
        return "%s %s, %s %s" % (dollars, dollar_unit, cents, cent_unit)
    elif dollars:
        dollar_unit = "dollar" if dollars == 1 else "dollars"
        return "%s %s" % (dollars, dollar_unit)
    elif cents:
        cent_unit = "cent" if cents == 1 else "cents"
        return "%s %s" % (cents, cent_unit)
    else:
        return "zero dollars"
def _expand_ordinal(m):
    return _inflect.number_to_words(m.group(0))
def _expand_number(m):
    num = int(m.group(0))
    if num > 1000 and num < 3000:
        if num == 2000:
            return "two thousand"
        elif num > 2000 and num < 2010:
            return "two thousand " + _inflect.number_to_words(num % 100)
        elif num % 100 == 0:
            return _inflect.number_to_words(num // 100) + " hundred"
        else:
            return _inflect.number_to_words(num, andword="", zero="oh", group=2).replace(", ", " ")
    else:
        return _inflect.number_to_words(num, andword="")
def normalize_numbers(text):
    text = re.sub(_comma_number_re, _remove_commas, text)
    text = re.sub(_pounds_re, r"\1 pounds", text)
    text = re.sub(_dollars_re, _expand_dollars, text)
    text = re.sub(_decimal_number_re, _expand_decimal_point, text)
    text = re.sub(_ordinal_re, _expand_ordinal, text)
    text = re.sub(_number_re, _expand_number, text)
    return text


##text synthesizer

_symbol_to_id = {s: i for i, s in enumerate(symbols)}
_id_to_symbol = {i: s for i, s in enumerate(symbols)}

_curly_re = re.compile(r"(.*?)\{(.+?)\}(.*)")

def text_to_sequence(text, cleaner_names):

    sequence = []

    while len(text):
        m = _curly_re.match(text)
        if not m:
            sequence += _symbols_to_sequence(_clean_text(text, cleaner_names))
            break
        sequence += _symbols_to_sequence(_clean_text(m.group(1), cleaner_names))
        sequence += _arpabet_to_sequence(m.group(2))
        text = m.group(3)

    # Append EOS token
    sequence.append(_symbol_to_id["~"])
    return sequence
def sequence_to_text(sequence):
    # convierte una sequencia de simbolos a texto
    result = ""
    for symbol_id in sequence:
        if symbol_id in _id_to_symbol:
            s = _id_to_symbol[symbol_id]

            if len(s) > 1 and s[0] == "@":
                s = "{%s}" % s[1:]
            result += s
    return result.replace("}{", " ")
def _clean_text(text, cleaner_names):
    for name in cleaner_names:
        cleaner = getattr(cleaners, name)
        if not cleaner:
            raise Exception("Unknown cleaner: %s" % name)
        text = cleaner(text)
    return text
def _symbols_to_sequence(symbols):
    return [_symbol_to_id[s] for s in symbols if _should_keep_symbol(s)]
def _arpabet_to_sequence(text):
    return _symbols_to_sequence(["@" + s for s in text.split()])
def _should_keep_symbol(s):
    return s in _symbol_to_id and s not in ("_", "~")


#SYnthesizer
class Synthesizer:
    sample_rate = hparams.sample_rate
    hparams = hparams

    def __init__(self, model_fpath: Path, verbose=True):

        self.model_fpath = model_fpath
        self.verbose = verbose

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        self._model = None

    def is_loaded(self):
        return self._model is not None

    def load(self):

        self._model = Tacotron(embed_dims=hparams.tts_embed_dims,
                               num_chars=len(symbols),
                               encoder_dims=hparams.tts_encoder_dims,
                               decoder_dims=hparams.tts_decoder_dims,
                               n_mels=hparams.num_mels,
                               fft_bins=hparams.num_mels,
                               postnet_dims=hparams.tts_postnet_dims,
                               encoder_K=hparams.tts_encoder_K,
                               lstm_dims=hparams.tts_lstm_dims,
                               postnet_K=hparams.tts_postnet_K,
                               num_highways=hparams.tts_num_highways,
                               dropout=hparams.tts_dropout,
                               stop_threshold=hparams.tts_stop_threshold,
                               speaker_embedding_size=hparams.speaker_embedding_size).to(self.device)

        self._model.load(self.model_fpath)
        self._model.eval()

    def synthesize_spectrograms(self, texts: List[str],
                                embeddings: Union[np.ndarray, List[np.ndarray]],
                                return_alignments=False):

        if not self.is_loaded():
            self.load()

        inputs = [text_to_sequence(text.strip(), hparams.tts_cleaner_names) for text in texts]
        if not isinstance(embeddings, list):
            embeddings = [embeddings]

        batched_inputs = [inputs[i:i+hparams.synthesis_batch_size]
                             for i in range(0, len(inputs), hparams.synthesis_batch_size)]
        batched_embeds = [embeddings[i:i+hparams.synthesis_batch_size]
                             for i in range(0, len(embeddings), hparams.synthesis_batch_size)]

        specs = []
        for i, batch in enumerate(batched_inputs, 1):
            text_lens = [len(text) for text in batch]
            max_text_len = max(text_lens)
            chars = [pad1d(text, max_text_len) for text in batch]
            chars = np.stack(chars)

            speaker_embeds = np.stack(batched_embeds[i-1])

            chars = torch.tensor(chars).long().to(self.device)
            speaker_embeddings = torch.tensor(speaker_embeds).float().to(self.device)

            _, mels, alignments = self._model.generate(chars, speaker_embeddings)
            mels = mels.detach().cpu().numpy()
            for m in mels:
                while np.max(m[:, -1]) < hparams.tts_stop_threshold:
                    m = m[:, :-1]
                specs.append(m)

        return (specs, alignments) if return_alignments else specs

    @staticmethod
    def load_preprocess_wav(fpath):
        wav = librosa.load(str(fpath), hparams.sample_rate)[0]
        if hparams.rescale:
            wav = wav / np.abs(wav).max() * hparams.rescaling_max
        return wav

    @staticmethod
    def make_spectrogram(fpath_or_wav: Union[str, Path, np.ndarray]):
        if isinstance(fpath_or_wav, str) or isinstance(fpath_or_wav, Path):
            wav = Synthesizer.load_preprocess_wav(fpath_or_wav)
        else:
            wav = fpath_or_wav

        mel_spectrogram = audio.melspectrogram(wav, hparams).astype(np.float32)
        return mel_spectrogram

    @staticmethod
    def griffin_lim(mel):
        return audio.inv_mel_spectrogram(mel, hparams)

def pad1d(x, max_len, pad_value=0):
    return np.pad(x, (0, max_len - len(x)), mode="constant", constant_values=pad_value)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("-e", "--enc_model_fpath", type=Path,
                        default="saved_models/default/encoder.pt",
                        help="Path to a saved encoder")
    parser.add_argument("-s", "--syn_model_fpath", type=Path,
                        default="saved_models/default/synthesizer.pt",
                        help="Path to a saved synthesizer")
    parser.add_argument("-v", "--voc_model_fpath", type=Path,
                        default="saved_models/default/vocoder.pt",
                        help="Path to a saved vocoder")
    parser.add_argument("--cpu", action="store_true", help=\
        "Si es True, el procesamiento se realiza en la CPU, incluso cuando hay una GPU disponible.")
    parser.add_argument("--no_sound", action="store_true", help=\
        "Si es True, el audio no se reproducirá.")
    parser.add_argument("--seed", type=int, default=None, help=\
        "Valor opcional de la semilla de numeros aleatorios para que la caja de herramientas sea determinista.")
    args = parser.parse_args()
    arg_dict = vars(args)
    print_args(args, parser)

    # Determinar si se debe usar la CPU o la GPU
    if arg_dict.pop("cpu"):
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    if torch.cuda.is_available(): # si hay una gpu disponible
        device_id = torch.cuda.current_device() # se obtiene el id de la gpu
        gpu_properties = torch.cuda.get_device_properties(device_id)  # se obtienen las propiedades de la gpu
        ## Print some environment information (for debugging purposes)
        print("Se encontró %d GPU disponible. Usando GPU %d (%s) de capacidad %d.%d con "
            "%.1fGb de memoria total.\n" %
            (torch.cuda.device_count(),
            device_id,
            gpu_properties.name,
            gpu_properties.major,
            gpu_properties.minor,
            gpu_properties.total_memory / 1e9)) # se imprimen las propiedades de la gpu

    #cargar los modelos
    print("Preparando encoder, synthesizer y vocoder...")

    ensure_default_models(Path("saved_models"))# si no existen los modelos, se descargan

    # se pasan los parametros de los modelos
    encoder.load_model(args.enc_model_fpath)
    synthesizer = Synthesizer(args.syn_model_fpath)
    vocoder.load_model(args.voc_model_fpath)


    print("\tProbando encoder...")
    encoder.embed_utterance(np.zeros(encoder.sampling_rate))


    embed = np.random.rand(speaker_embedding_size) # se genera un embedding aleatorio

    embed /= np.linalg.norm(embed) # se normaliza el embedding

    embeds = [embed, np.zeros(speaker_embedding_size)] # se crea una lista de embeddings
    texts = ["test 1", "test 2"] # se crea una lista de textos
    print("\tProbando synthesizer...") # se prueba el synthesizer
    mels = synthesizer.synthesize_spectrograms(texts, embeds) # se sintetizan los mels


    mel = np.concatenate(mels, axis=1) # unir los mels

    no_action = lambda *args: None
    print("\tProbando vocoder...")

    vocoder.infer_waveform(mel, target=200, overlap=50, progress_callback=no_action) # se prueba el vocoder

    print("Todas las pruebas pasaron. Ahora podemos sintetizar texto.\n")

    num_generated = 0 # contador de archivos generados
    while True:
        try:
            # Ruta de audio
            message = "Ingresar ruta de audio (mp3, " \
                      "wav, m4a, flac, ...):\n"
            in_fpath = Path(input(message).replace("\"", "").replace("\'", ""))

            # Computando el embedding
            # Primero, cargamos el wav usando la funcion que ofrece el encoder

            # Los metodos siguientes son equivalentes:
            # - Cargarlo directamente de la ruta:
            preprocessed_wav = encoder.preprocess_wav(in_fpath)
            # - Una vez cargado:
            original_wav, sampling_rate = librosa.load(str(in_fpath))
            # Crea matriz np
            preprocessed_wav = encoder.preprocess_wav(original_wav, sampling_rate)
            # Eliminamos silencio usando periodos. Normalizar. Ventana.
            print("Correcto preprocesamiento.")


            # Fragmenta en frames los elementos del vector.
            embed = encoder.embed_utterance(preprocessed_wav)


            # Input
            text = input("Escribir una oración para sintetizar:\n")

            # Forzamos el reload del sintetizador
            if args.seed is not None:
                torch.manual_seed(args.seed)
                synthesizer = Synthesizer(args.syn_model_fpath)

            # El sintetizador funciona por lotes, por lo que debe colocar sus datos en una lista o matriz numérica
            texts = [text] 
            embeds = [embed]
            # Pasamos el texto y el vector de incrustacion al sintetizador
            specs = synthesizer.synthesize_spectrograms(texts, embeds)
            spec = specs[0]
            # Se produce el espectograma usando el sintetizador (tacotron)

            print("Espectograma creado.")

            ## Pasamos el espectograma de mel al vocoder que transforma esas senales en audio
            print("Sintetizando el Waveform:")

            # Recargamos el vocoder
            if args.seed is not None:
                torch.manual_seed(args.seed)
                vocoder.load_model(args.voc_model_fpath)

            # Sintetizar la forma de onda es bastante sencillo.
            generated_wav = vocoder.infer_waveform(spec)


            # Correccion de error, contar el ultimo segundo
            generated_wav = np.pad(generated_wav, (0, synthesizer.sample_rate), mode="constant")

            # Correccion de error, recorte el exceso de silencios para compensar las lagunas en los espectrogramas
            generated_wav = encoder.preprocess_wav(generated_wav)

            # Lo guardamos en el disco
            filename = "demo_output_%02d.wav" % num_generated

            sf.write(filename, generated_wav.astype(np.float32), synthesizer.sample_rate)
            num_generated += 1

            print("\nOutput guardado en %s\n\n" % filename)


        except Exception as e:
            print("Excepcion: %s" % repr(e))
            print("Resiniciando\n")
