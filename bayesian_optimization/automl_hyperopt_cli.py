"""CLI de AutoML4NILM: entrena y evalua los 13 algoritmos NILM.

Dos modos, siempre con el algoritmo fijo (hyperopt ya no elige el algoritmo):

- barrido:   una corrida por algoritmo con los hiperparametros por defecto.
- optimizar: optimizacion bayesiana (TPE) de los hiperparametros de cada
             algoritmo, --max_evals evaluaciones por algoritmo.

Cada corrida (OK o fallida) se agrega como una linea JSON al archivo --salida.
La metrica a optimizar se calcula sobre la casa de validacion; las metricas de
la casa de test se guardan para comparar la generalizacion entre casas.

Ejemplo (desde bayesian_optimization/):
    python automl_hyperopt_cli.py --datapath ../../data/ukdale.h5 \\
        --appliance "fridge freezer" --sampling_rate 60 \\
        --train_building 1 --train_start 2014-03-01 --train_end 2014-03-06 \\
        --val_building 5 --val_start 2014-07-01 --val_end 2014-07-02 \\
        --test_building 2 --test_start 2013-06-01 --test_end 2013-06-03 \\
        --epochs 1 --modo barrido --algoritmos seq2point sgn \\
        --salida results/barrido.jsonl
"""
import warnings; warnings.filterwarnings("ignore")
import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import argparse
import datetime
import json
import logging
import time
import traceback

import numpy as np
from hyperopt import fmin, tpe, hp, STATUS_OK, STATUS_FAIL, Trials, space_eval

from nilmtk.appliance import Appliance
# Busqueda de aparatos por tipo exacto. Con sinonimos (el default), NILMTK
# trata fridge / fridge freezer / freezer como el mismo aparato: en una casa
# con heladera y freezer aparte la busqueda es ambigua, y en una con solo un
# freezer devuelve el freezer como si fuera la heladera. Las etiquetas se
# unifican antes con models/nilm/normalizar_aparatos.py (repo AMPR).
Appliance.allow_synonyms = False

import tables
import tensorflow as tf
from tensorflow.keras.optimizers import Adam, Nadam, RMSprop

from algorithms.randomforest import random_forest
from algorithms.dt import decision_tree
from algorithms.dae import dae
from algorithms.fcnn import fcnn
from algorithms.fhmm import fhmm
from algorithms.co import combinatorial_optimisation
from algorithms.gru import gru
from algorithms.seq2point import seq2point
from algorithms.treecnn import treecnn
from algorithms.sgn import sgn
from algorithms.seq2seq import seq2seq
from algorithms.windowgru import window_gru
from algorithms.lstm import lstm

logger = logging.getLogger("automl4nilm")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

#######################################################
################## Algoritmos e hiperparametros
#######################################################
# Nombre en el CLI -> (funcion, hiperparametros que recibe)
NN = ['optimizer', 'learning_rate', 'loss']
ALGORITMOS = {
    'dae':                             (dae,         NN + ['sequence_length']),
    'fully-connected neural networks': (fcnn,        NN + ['num_layers', 'dropout_prob']),
    'gated recurrent units':           (gru,         NN),
    'window gru':                      (window_gru,  NN + ['window_size']),
    'seq2seq':                         (seq2seq,     NN + ['window_size']),
    'seq2point':                       (seq2point,   NN + ['window_size']),
    'treecnn':                         (treecnn,     NN + ['window_size', 'kernel_size']),
    'sgn':                             (sgn,         NN + ['window_size']),
    'long short-term memory':          (lstm,        NN),
    'decision tree':                   (decision_tree, ['criterion', 'min_samples_split']),
    'random forest':                   (random_forest, ['n_estimators', 'criterion', 'min_samples_split']),
    'combinatorial optimization':      (combinatorial_optimisation, []),
    'factorial hidden markov models':  (fhmm,        []),
}
# Los que entrenan por epocas (reciben num_epochs y patience)
CON_EPOCAS = {a for a, (_, hps) in ALGORITMOS.items() if 'optimizer' in hps}

# Hiperparametros por defecto (modo barrido)
DEFAULTS = {
    'optimizer': 'adam',
    'learning_rate': 0.001,
    'loss': 'mse',
    'window_size': 99,
    'sequence_length': 50,
    'num_layers': 5,
    'dropout_prob': 0.1,
    'criterion': 'squared_error',
    'min_samples_split': 10,
    'n_estimators': 30,
    'kernel_size': 7,  # TreeCNN: el del paper
}
# Ventana por defecto especifica de cada algoritmo (la de 99 es la de seq2point)
DEFAULTS_POR_ALGORITMO = {
    'window gru': {'window_size': 50},
}

# Espacio de busqueda (modo optimizar)
ESPACIO = {
    'optimizer': hp.choice('optimizer', ['adam', 'nadam', 'rmsprop']),
    'learning_rate': hp.choice('learning_rate', [0.0001, 0.0003, 0.001, 0.003]),
    'loss': hp.choice('loss', ['mse', 'mae']),
    'window_size': hp.choice('window_size', [49, 99, 199, 299]),
    'sequence_length': hp.choice('sequence_length', [20, 50, 100]),
    'num_layers': hp.choice('num_layers', [3, 5, 7]),
    'dropout_prob': hp.choice('dropout_prob', [0.1, 0.3, 0.5]),
    'criterion': hp.choice('criterion', ['squared_error', 'friedman_mse']),
    'min_samples_split': hp.choice('min_samples_split', [2, 10, 20, 50]),
    'n_estimators': hp.choice('n_estimators', [10, 30, 50, 100]),
    'kernel_size': hp.choice('kernel_size', [7, 15, 31]),
}
ESPACIO_POR_ALGORITMO = {
    'window gru': {'window_size': hp.choice('window_size', [20, 50, 100])},
}

OPTIMIZADORES = {'adam': Adam, 'nadam': Nadam, 'rmsprop': RMSprop}

# Algoritmos que aceptan varias casas de train y normalizacion fija
# (algorithms/multi_casa.py). Potencia tipica del aparato para la fija (W).
MULTI_CASA = {'seq2point', 'sgn', 'treecnn', 'seq2seq'}
POTENCIA_APARATO = {'fridge freezer': 300.0, 'washing machine': 2500.0, 'microwave': 3000.0}

# Metricas a maximizar (hyperopt minimiza: se invierten)
A_MAXIMIZAR = {'precision_score', 'recall_score', 'accuracy_score', 'f1_score', 'disaggregation_accuracy'}


def a_serializable(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def cerrar_hdf_abiertos():
    """Cierra los HDF5 que un algoritmo fallido dejo abiertos.

    Varios algoritmos comparten el nombre del archivo de desagregacion
    (p. ej. disag-out-val.h5): si uno falla con el archivo abierto, el
    siguiente no puede abrirlo en modo 'w'. Solo se cierran esos: NILMTK
    tiene sus propios HDF5 temporales abiertos y cerrarlos rompe las
    corridas siguientes.
    """
    for h in list(tables.file._open_files.handlers):
        if os.path.basename(h.filename).startswith('disag-'):
            h.close()


#######################################################
################## Una corrida
#######################################################
def entrenar(algoritmo, hparams, cfg):
    """Entrena y evalua un algoritmo. Devuelve el registro para --salida."""
    funcion, _ = ALGORITMOS[algoritmo]
    kwargs = dict(
        dataset_path=cfg.datapath,
        train_building=cfg.train_building, train_start=cfg.train_start, train_end=cfg.train_end,
        val_building=cfg.val_building, val_start=cfg.val_start, val_end=cfg.val_end,
        test_building=cfg.test_building, test_start=cfg.test_start, test_end=cfg.test_end,
        meter_key=cfg.appliance,
        sample_period=cfg.sampling_rate,
    )
    if algoritmo in MULTI_CASA:
        kwargs.update(train_extra=cfg.train_extra, normalizacion=cfg.normalizacion,
                      potencia_aparato=POTENCIA_APARATO.get(cfg.appliance))
    for k, v in hparams.items():
        if k == 'optimizer':
            # Instancia con el learning rate: pasando el nombre ('adam'),
            # Keras usa su learning rate por defecto e ignora el elegido.
            v = OPTIMIZADORES[v](learning_rate=hparams['learning_rate'])
        kwargs[k] = v
    if algoritmo in CON_EPOCAS:
        kwargs['num_epochs'] = cfg.epochs
        kwargs['patience'] = cfg.patience

    registro = {
        'fecha': datetime.datetime.now().isoformat(timespec='seconds'),
        'modo': cfg.modo,
        'algorithm': algoritmo,
        'hparams': hparams,
        'datapath': os.path.basename(cfg.datapath),
        'appliance': cfg.appliance,
        'sampling_rate': cfg.sampling_rate,
        'train': [cfg.train_building, cfg.train_start, cfg.train_end],
        'train_extra': [list(t) for t in cfg.train_extra],
        'normalizacion': cfg.normalizacion,
        'val': [cfg.val_building, cfg.val_start, cfg.val_end],
        'test': [cfg.test_building, cfg.test_start, cfg.test_end],
        'max_epochs': cfg.epochs if algoritmo in CON_EPOCAS else None,
        'seed': cfg.seed,
    }
    if cfg.seed is not None:
        tf.keras.utils.set_random_seed(cfg.seed)

    inicio = time.time()
    try:
        r = funcion(**kwargs)
    except Exception as e:
        traceback.print_exc()
        cerrar_hdf_abiertos()
        registro.update(status=STATUS_FAIL, error=f"{type(e).__name__}: {e}",
                        time_taken=round(time.time() - inicio, 2))
    else:
        registro.update(status=STATUS_OK,
                        val_metrics=r['val_metrics'],
                        test_metrics=r['test_metrics'],
                        epochs=r['epochs'],
                        time_taken=round(time.time() - inicio, 2))
    finally:
        tf.keras.backend.clear_session()

    with open(cfg.salida, 'a') as f:
        f.write(json.dumps(registro, default=a_serializable) + '\n')
    return registro


def valor_a_minimizar(registro, metrica):
    v = float(registro['val_metrics'][metrica])
    return -v if metrica in A_MAXIMIZAR else v


#######################################################
################## Modos
#######################################################
def hparams_por_defecto(algoritmo):
    _, nombres = ALGORITMOS[algoritmo]
    base = {**DEFAULTS, **DEFAULTS_POR_ALGORITMO.get(algoritmo, {})}
    return {k: base[k] for k in nombres}


def barrido(cfg):
    for algoritmo in cfg.algoritmos:
        hparams = hparams_por_defecto(algoritmo)
        logger.info(f"[barrido] {algoritmo} {hparams}")
        r = entrenar(algoritmo, hparams, cfg)
        if r['status'] == STATUS_OK:
            logger.info(f"[barrido] {algoritmo}: {cfg.metrica} val={r['val_metrics'][cfg.metrica]:.4f} "
                        f"test={r['test_metrics'][cfg.metrica]:.4f} ({r['time_taken']} s)")
        else:
            logger.error(f"[barrido] {algoritmo} FALLO: {r['error']}")


def optimizar(cfg):
    for algoritmo in cfg.algoritmos:
        _, nombres = ALGORITMOS[algoritmo]
        if not nombres:
            # CO y FHMM no tienen hiperparametros: una corrida alcanza
            logger.info(f"[optimizar] {algoritmo} sin hiperparametros: una sola corrida")
            entrenar(algoritmo, {}, cfg)
            continue

        base = {**ESPACIO, **ESPACIO_POR_ALGORITMO.get(algoritmo, {})}
        espacio = {k: base[k] for k in nombres}

        def objetivo(hparams):
            hparams = {k: a_serializable(v) if isinstance(v, np.generic) else v for k, v in hparams.items()}
            r = entrenar(algoritmo, hparams, cfg)
            if r['status'] != STATUS_OK:
                return {'status': STATUS_FAIL}
            logger.info(f"[optimizar] {algoritmo} {hparams}: {cfg.metrica} val={r['val_metrics'][cfg.metrica]:.4f}")
            return {'status': STATUS_OK, 'loss': valor_a_minimizar(r, cfg.metrica)}

        trials = Trials()
        rstate = np.random.default_rng(cfg.seed) if cfg.seed is not None else None
        try:
            mejor = fmin(fn=objetivo, space=espacio, algo=tpe.suggest,
                         max_evals=cfg.max_evals, trials=trials, rstate=rstate,
                         # La barra de tqdm reemplaza stdout y rompe la de Keras
                         show_progressbar=False)
            logger.info(f"[optimizar] {algoritmo} mejor: {space_eval(espacio, mejor)}")
        except Exception as e:
            # fmin falla si todas las evaluaciones fallaron: seguir con el proximo
            logger.error(f"[optimizar] {algoritmo} sin evaluaciones validas: {e}")


#######################################################
################## CLI
#######################################################
def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--datapath', required=True, help='Dataset NILMTK (.h5) ya normalizado')
    p.add_argument('--appliance', required=True, help='Tipo exacto de NILMTK, p. ej. "fridge freezer"')
    p.add_argument('--sampling_rate', type=int, default=60, help='Segundos (default 60)')
    for parte in ('train', 'val', 'test'):
        p.add_argument(f'--{parte}_building', type=int, required=True)
        p.add_argument(f'--{parte}_start', required=True)
        p.add_argument(f'--{parte}_end', required=True)
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--patience', type=int, default=5)
    p.add_argument('--algoritmos', nargs='+', default=['todos'],
                   help='Nombres del CLI (entre comillas si tienen espacios) o "todos"')
    p.add_argument('--modo', choices=['barrido', 'optimizar'], default='barrido')
    p.add_argument('--max_evals', type=int, default=10, help='Evaluaciones por algoritmo (modo optimizar)')
    p.add_argument('--metrica', default='mean_absolute_error', help='Metrica de validacion a optimizar')
    p.add_argument('--seed', type=int, default=42,
                   help='Semilla de Python, numpy y TF (y de hyperopt). Default 42')
    p.add_argument('--salida', default='results/trials.jsonl', help='JSONL donde se agrega cada corrida')
    p.add_argument('--train_extra', nargs='*', default=[], metavar='CASA:INICIO:FIN',
                   help=f'Casas de train adicionales (solo {sorted(MULTI_CASA)}), p. ej. 2:2014-05-01:2014-06-30')
    p.add_argument('--normalizacion', choices=['pico', 'fija'], default='pico',
                   help='pico: / maximo del agregado (original); fija: agregado estandarizado y aparato / potencia tipica')
    cfg = p.parse_args(argv)

    if cfg.algoritmos == ['todos']:
        cfg.algoritmos = list(ALGORITMOS)
    desconocidos = [a for a in cfg.algoritmos if a not in ALGORITMOS]
    if desconocidos:
        p.error(f"algoritmos desconocidos: {desconocidos}. Validos: {list(ALGORITMOS)}")
    try:
        cfg.train_extra = [(int(b), ini, fin) for b, ini, fin in (t.split(':') for t in cfg.train_extra)]
    except ValueError:
        p.error("--train_extra: usar CASA:INICIO:FIN, p. ej. 2:2014-05-01:2014-06-30")
    if cfg.train_extra or cfg.normalizacion != 'pico':
        otros = [a for a in cfg.algoritmos if a not in MULTI_CASA]
        if otros:
            p.error(f"--train_extra / --normalizacion fija solo para {sorted(MULTI_CASA)}; no para {otros}")
        if cfg.normalizacion == 'fija' and cfg.appliance not in POTENCIA_APARATO:
            p.error(f"sin potencia tipica para '{cfg.appliance}' (POTENCIA_APARATO)")
    if cfg.test_building == cfg.train_building:
        logger.warning("La casa de test es la de train: no mide generalizacion entre casas")
    os.makedirs(os.path.dirname(os.path.abspath(cfg.salida)), exist_ok=True)
    return cfg


def main(argv=None):
    cfg = parse_args(argv)
    logger.info(f"Python {sys.version.split()[0]}, TF {tf.__version__}, GPUs: {tf.config.list_physical_devices('GPU')}")
    logger.info(f"Modo {cfg.modo}: {len(cfg.algoritmos)} algoritmos -> {cfg.salida}")
    (barrido if cfg.modo == 'barrido' else optimizar)(cfg)


if __name__ == "__main__":
    main()
