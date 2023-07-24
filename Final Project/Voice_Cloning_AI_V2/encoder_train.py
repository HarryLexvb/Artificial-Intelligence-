from utils.argutils import print_args # Funcion para imprimir los argumentos
from encoder.train import train # Funcion para entrenar el modelo
from pathlib import Path # Es una biblioteca para trabajar con rutas de archivos y directorios
import argparse # es biblioteca es para analizar argumentos de la linea de comandos


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Entrena el codificador del altavoz. Debe haber ejecutado primero encoder_preprocess.py.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("run_id", type=str, help= \
        "Nombre para este modelo. Por defecto, los resultados del entrenamiento se almacenaran en saved_models/<run_id>/. Si un estado de modelo "
        "del mismo ID de ejecución fue guardado previamente, el entrenamiento se reiniciara desde ahi. Pase -f para sobrescribir "
        "estados guardados y reiniciar desde cero.")
    parser.add_argument("clean_data_root", type=Path, help= \
        "Ruta al directorio de salida de encoder_preprocess.py. Si dejaste el directorio de  "
        "directorio de salida al preprocesar, debería ser <datasets_root>/SV2TTS/encoder/.")
    parser.add_argument("-m", "--models_dir", type=Path, default="saved_models", help=\
        "Ruta al directorio raiz que contiene todos los modelos. Se creara un directorio <run_name> bajo esta raiz "
        "Contendra los pesos guardados del modelo, asi como copias de seguridad de dichos pesos y graficos generados durante "
        "el entrenamiento.")
    parser.add_argument("-v", "--vis_every", type=int, default=10, help= \
        "Numero de pasos entre las actualizaciones de la pérdida y las parcelas.")
    parser.add_argument("-u", "--umap_every", type=int, default=100, help= \
        "Numero de pasos entre actualizaciones de la proyección umap. Establézcalo a 0 para no actualizar nunca las "
        "proyecciones.")
    parser.add_argument("-s", "--save_every", type=int, default=500, help= \
        "Numero de pasos entre actualizaciones del modelo en el disco. Establezcalo a 0 para no guardar nunca el "
        "modelo.")
    parser.add_argument("-b", "--backup_every", type=int, default=7500, help= \
        "Número de pasos entre copias de seguridad del modelo. Establézcalo en 0 para no hacer nunca copias de seguridad del "
        "modelo.")
    parser.add_argument("-f", "--force_restart", action="store_true", help= \
        "No cargue ningun modelo guardado.")
    parser.add_argument("--visdom_server", type=str, default="http://localhost")
    parser.add_argument("--no_visdom", action="store_true", help= \
        "Disable visdom.")
    args = parser.parse_args()

    # Run the training
    print_args(args, parser)
    train(**vars(args)) # Entrena el modelo