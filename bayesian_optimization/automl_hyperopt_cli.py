import warnings; warnings.filterwarnings("ignore")
from hyperopt import fmin, tpe, hp, STATUS_OK, STATUS_FAIL, Trials, space_eval
import os, sys
sys.path.append(os.path.abspath('./bayesian_optimization/'))
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

# For surpressing print

print (sys.version)
#Using HiddenPrints is useful in scenarios where you want to run code that produces a lot of output (like logging or progress messages) but you don't want that output to clutter your console.


from nilmtk import DataSet
from nilmtk.appliance import Appliance
# Busqueda de aparatos por tipo exacto. Con sinonimos (el default), NILMTK
# trata fridge / fridge freezer / freezer como el mismo aparato: en una casa
# con heladera y freezer aparte la busqueda es ambigua, y en una con solo un
# freezer devuelve el freezer como si fuera la heladera. Las etiquetas se
# unifican antes con models/nilm/normalizar_aparatos.py (repo AMPR).
Appliance.allow_synonyms = False
import pandas as pd
import numpy as np

from tensorflow.keras.optimizers import Adam, Nadam, RMSprop

import datetime
import time
import math
import glob

from sklearn.tree import DecisionTreeRegressor

import argparse

import json

# Import algorithms
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

from utils import metrics_np
from utils.metrics_np import Metrics

#######################################################
################## Function for reversing metrics for minimization
#######################################################
#metrics_minmax_reverse: Used when working with a collection of metric results (e.g., from model evaluation).
#metrics_minmax_reverse_print: Used for individual metric values, especially when preparing output for logging or reporting.
def metrics_minmax_reverse(metric_results, metrics_to_optimize):
    metric_to_reverse = ['precision_score', 'accuracy_score', 'f1_score', 'disaggregation_accuracy']
    if metrics_to_optimize in metric_to_reverse:
        # return negative to maximize a metric instead
        return -1*metric_results[metrics_to_optimize]
    else:
        return metric_results[metrics_to_optimize]


#The function metrics_minmax_reverse_print is designed to adjust the output of specific performance metrics based on whether they need to be maximized or not.
def metrics_minmax_reverse_print(metric, metrics_to_optimize):
    metric_to_reverse = ['precision_score', 'accuracy_score', 'f1_score', 'disaggregation_accuracy']
    if metrics_to_optimize in metric_to_reverse:
        # return negative to maximize a metric instead
        return -1*metric
    else:
        return metric

#######################################################
################## Global Variables
#######################################################
count = 0
best = 0

hd5_filepath = None

train_building = None
train_start = None
train_end = None

val_building = None
val_start = None
val_end = None

test_building = None
test_start = None
test_end = None
appliance = None
downsampling_period = None

metrics_to_optimize = None
num_epochs = None
patience = None

#######################################################
################## Main objective function for hyperopt
#######################################################
def to_serializable(obj):
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

import logging
logger = logging.getLogger()
logging.basicConfig(level=logging.INFO)

# Objective Function
def objective(args):
    global best, count, num_epochs, patience, metrics_to_optimize
    count += 1
    algorithm = args['algorithm']
    args['type'] = algorithm


    #algorithm = args['type']

    try:
        algorithm = args['algorithm']
        logger.info(f"Running optimization for algorithm: {algorithm}")

        dataset_path = args['datapath']
        train_building = args['train_building']
        train_start = args['train_start']
        train_end = args['train_end']
        val_building = args['val_building']
        val_start = args['val_start']
        val_end = args['val_end']
        test_building = args['test_building']
        test_start = args['test_start']
        test_end = args['test_end']
        appliance = args['appliance']
        patience = args['patience']
        num_epochs = args['num_epochs']
        downsampling_period = args['sampling_rate']

        if algorithm == 'dae':
            model_result_data = dae(
                dataset_path=dataset_path,
                train_building=train_building, train_start=train_start, train_end=train_end,
                val_building=val_building, val_start=val_start, val_end=val_end,
                test_building=test_building, test_start=test_start, test_end=test_end,
                meter_key=appliance,
                sample_period=downsampling_period,
                optimizer=args['optimizer'],
                learning_rate=args['learning_rate'],
                loss=args['loss_function'],
                patience=patience,
                num_epochs=num_epochs,
                sequence_length=args['sequence_length']
            )
        elif algorithm == 'fully-connected neural networks':
            model_result_data = fcnn(
                dataset_path=dataset_path,
                train_building=train_building, train_start=train_start, train_end=train_end,
                val_building=val_building, val_start=val_start, val_end=val_end,
                test_building=test_building, test_start=test_start, test_end=test_end,
                meter_key=appliance,
                sample_period=downsampling_period,
                num_epochs=num_epochs,
                patience=patience,
                num_layers=int(args['num_layers']),
                optimizer=args['optimizer'],
                learning_rate=args['learning_rate'],
                dropout_prob=args['dropout_prob'],
                loss=args['loss_function']
            )
        elif algorithm == 'combinatorial optimization':
            model_result_data = combinatorial_optimisation(
                dataset_path=hd5_filepath,
                train_building=train_building, train_start=train_start, train_end=train_end,
                val_building=val_building, val_start=val_start, val_end=val_end,
                test_building=test_building, test_start=test_start, test_end=test_end,
                meter_key=appliance,
                sample_period=downsampling_period)

        elif algorithm == 'factorial hidden markov models':
            model_result_data = fhmm(
                dataset_path=hd5_filepath,
                train_building=train_building, train_start=train_start, train_end=train_end,
                val_building=val_building, val_start=val_start, val_end=val_end,
                test_building=test_building, test_start=test_start, test_end=test_end,
                meter_key=appliance,
                sample_period=downsampling_period)

        elif algorithm == 'gated recurrent units':
            model_result_data = gru(
                dataset_path=dataset_path,
                train_building=train_building, train_start=train_start, train_end=train_end,
                val_building=val_building, val_start=val_start, val_end=val_end,
                test_building=test_building, test_start=test_start, test_end=test_end,
                meter_key=appliance,
                sample_period=downsampling_period,
                num_epochs=num_epochs,
                patience=patience,
                optimizer=args['optimizer'],
                learning_rate=args['learning_rate'],
                loss=args['loss_function']
            )
        elif algorithm == 'window gru':
            model_result_data = window_gru(
                dataset_path=dataset_path,
                train_building=train_building, train_start=train_start, train_end=train_end,
                val_building=val_building, val_start=val_start, val_end=val_end,
                test_building=test_building, test_start=test_start, test_end=test_end,
                meter_key=appliance,
                sample_period=downsampling_period,
                num_epochs=num_epochs,
                patience=patience,
                optimizer=args['optimizer'],
                learning_rate=args['learning_rate'],
                loss=args['loss_function'],
                window_size=args['window_size']
            )
        elif algorithm == 'seq2seq':
            model_result_data = seq2seq(
                dataset_path=dataset_path,
                train_building=train_building, train_start=train_start, train_end=train_end,
                val_building=val_building, val_start=val_start, val_end=val_end,
                test_building=test_building, test_start=test_start, test_end=test_end,
                meter_key=appliance,
                sample_period=downsampling_period,
                num_epochs=num_epochs,
                patience=patience,
                optimizer=args['optimizer'],
                learning_rate=args['learning_rate'],
                loss=args['loss_function'],
                window_size=args['window_size'],
                sequence_length=args['sequence_length']
            )

        elif algorithm == 'seq2point':
            model_result_data = seq2point(
                dataset_path=dataset_path,
                train_building=train_building, train_start=train_start, train_end=train_end,
                val_building=val_building, val_start=val_start, val_end=val_end,
                test_building=test_building, test_start=test_start, test_end=test_end,
                meter_key=appliance,
                sample_period=downsampling_period,
                num_epochs=num_epochs,
                patience=patience,
                optimizer=args['optimizer'],
                learning_rate=args['learning_rate'],
                loss=args['loss_function'],
                window_size=args['window_size']
            )

        elif algorithm == 'sgn':
            model_result_data = sgn(
                dataset_path=dataset_path,
                train_building=train_building, train_start=train_start, train_end=train_end,
                val_building=val_building, val_start=val_start, val_end=val_end,
                test_building=test_building, test_start=test_start, test_end=test_end,
                meter_key=appliance,
                sample_period=downsampling_period,
                num_epochs=num_epochs,
                patience=patience,
                optimizer=args['optimizer'],
                learning_rate=args['learning_rate'],
                loss=args['loss_function'],
                window_size=args['window_size']
            )

        elif algorithm == 'treecnn':
            model_result_data = treecnn(
                dataset_path=dataset_path,
                train_building=train_building, train_start=train_start, train_end=train_end,
                val_building=val_building, val_start=val_start, val_end=val_end,
                test_building=test_building, test_start=test_start, test_end=test_end,
                meter_key=appliance,
                sample_period=downsampling_period,
                num_epochs=num_epochs,
                patience=patience,
                optimizer=args['optimizer'],
                learning_rate=args['learning_rate'],
                loss=args['loss_function'],
                window_size=args['window_size']
            )

        elif algorithm == 'long short-term memory':
            model_result_data = lstm(
                dataset_path=dataset_path,
                train_building=train_building, train_start=train_start, train_end=train_end,
                val_building=val_building, val_start=val_start, val_end=val_end,
                test_building=test_building, test_start=test_start, test_end=test_end,
                meter_key=appliance,
                sample_period=downsampling_period,
                num_epochs=num_epochs,
                patience=patience,
                optimizer=args['optimizer'],
                learning_rate=args['learning_rate'],
                loss=args['loss_function']
            )
        elif algorithm == 'decision tree':
            model_result_data = decision_tree(
                dataset_path=dataset_path,
                train_building=train_building, train_start=train_start, train_end=train_end,
                val_building=val_building, val_start=val_start, val_end=val_end,
                test_building=test_building, test_start=test_start, test_end=test_end,
                meter_key=appliance,
                sample_period=downsampling_period,
                criterion=args['criterion'],
                min_samples_split=args['min_samples_split']
            )
        elif algorithm == 'random forest':
            model_result_data = random_forest(
                dataset_path=dataset_path,
                train_building=train_building, train_start=train_start, train_end=train_end,
                val_building=val_building, val_start=val_start, val_end=val_end,
                test_building=test_building, test_start=test_start, test_end=test_end,
                meter_key=appliance,
                sample_period=downsampling_period,
                n_estimators=args['n_estimators'],
                criterion=args['criterion'],
                min_samples_split=args['min_samples_split']
            )


    except Exception as e:
        import traceback
        traceback.print_exc()  # This prints the full error stack trace

        if 'optimizer' in args:
            args['optimizer'] = str(args['optimizer'])

        results = {
            'args': args,
            'status': STATUS_FAIL,
            'error': str(e)
        }
        with open('results/trials_temp.json', 'a') as f:
            json.dump(results, f)
            f.write(os.linesep)
        return results

    # Convert keras optimizer type to String
    if 'optimizer' in args:
        args['optimizer'] = args['optimizer']

    # Extract info from model_result_data
    metrics = model_result_data['val_metrics'] # result of validation
    test_metrics = model_result_data['test_metrics']
    time_taken = model_result_data['time_taken']
    epochs = model_result_data['epochs']
    metrics_to_optimize = 'mean_absolute_error'

    # Print progress - Reguarly
    print ('iters:', count, ', ',metrics_to_optimize,':', metrics[metrics_to_optimize], 'using', args['type'])

    if count == 1:
        print('new best:', metrics[metrics_to_optimize], 'using', args['type'])
        best = metrics_minmax_reverse(metrics, metrics_to_optimize)
    elif metrics_minmax_reverse(metrics, metrics_to_optimize) < best:
        print ('new best:', metrics[metrics_to_optimize], 'using', args['type'])
        best = metrics_minmax_reverse(metrics, metrics_to_optimize)

    # Write trial result to file
    results = {
            'args': args,
            'loss': metrics[metrics_to_optimize], # return normall loss without need to inverse for maximizing
            'metrics': metrics,
            'test_metrics': test_metrics,
            'time_taken': time_taken,
            'epochs': epochs,
            'status': STATUS_OK,
            'order': count,
            }
    with open('results/trials_temp.json', 'a') as f:
        json.dump(results, f)
        f.write(os.linesep)

    return {
            'args': args,
            'loss': metrics_minmax_reverse(metrics, metrics_to_optimize), # Need to inverse for maximizing for fmin()
            'metrics': metrics,
            'test_metrics': test_metrics,
            'time_taken': time_taken,
            'epochs': epochs,
            'status': STATUS_OK,
            'order': count,
            }

    # Search space
    #.choice discrete options.
    #.quniform a continuous uniform distribution over a range
def optimize_hyperparameters():
        space = {
            'algorithm': hp.choice('algorithm', ['dae','fully-connected neural networks','gated recurrent units','window gru','seq2seq','seq2point','treecnn','sgn','long short-term memory','decision tree','random forest','combinatorial optimization','factorial hidden markov models']),
            'datapath': '../../data/ukdale.h5',
            'train_building': 1,
            'train_start': '2014-03-13',
            'train_end': '2014-07-21',
            'val_building': 1,
            'val_start': '2014-07-22',
            'val_end': '2014-07-30',
            'test_building': 1,
            'test_start': '2015-04-16',
            'test_end': '2015-05-15',
            'appliance': 'microwave',
            'sampling_rate': 60,
            'dropout_prob': hp.choice('dropout_prob', [0.1, 0.3]),
            'learning_rate': hp.choice('learning_rate', [0.0001, 0.001]),
            'num_layers': hp.choice('num_layers', [5, 7, 1]),
            'num_epochs': 3,
            'patience': 15,
            'criterion': hp.choice('criterion',['squared_error', 'friedman_mse']),
            'n_estimators':hp.choice('n_estimators',[10,30]),
            'window_size': hp.choice('window_size', [20, 50, 100]),
            'sequence_length': hp.choice('sequence_length', [10, 20, 50]),
            'min_samples_split': hp.choice('min_samples_split',[10,20]),
            'optimizer': hp.choice('optimizer', ['adam', 'nadam']),
            #'optimizer': hp.choice('optimizer', [ Adam, Nadam]),
            'loss_function': hp.choice('loss_function', ['mse', 'mae']),
        }

        trials = Trials()

        def wrapped_objective(args):
            return objective(args)

        # best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=10, trials=trials)
        best = fmin(fn=wrapped_objective, space=space, algo=tpe.suggest, max_evals=5, trials=trials)

        logger.info(f"Best hyperparameters: {best}")
        return best

    #######################################################
    ###### Start to Optimize
    #######################################################
def main():
        best_hyperparameters = optimize_hyperparameters()
        choices = {
            'algorithm': ['dae','fully-connected neural networks','gated recurrent units','window gru','seq2seq','seq2point','treecnn','sgn','long short-term memory','decision tree','random forest','combinatorial optimization','factorial hidden markov models'],
            'criterion': ['squared_error', 'friedman_mse'],
            'loss_function': ['mse', 'mae'],
            # 'max_depth': [10, 20, None],
            'num_layers': [5, 6, 7],
            'learning_rate': [0.0001, 0.01],
            'dropout_prob': [0.1, 0.6],
             'min_samples_split': [10,20],
             'n_estimators': [30,50],
            'optimizer': ['adam', 'nadam'],
            'window_size': [20, 50, 100],
            'sequence_length': [10, 20, 50],
        }

        # decoded = {key: choices[key][value] if key in choices else value for key, value in best_hyperparameters.items()}
        decoded = {}
        for key, value in best_hyperparameters.items():
            if key in choices:
                decoded[key] = choices[key][int(value)]
            elif isinstance(value, (np.integer,)):
                decoded[key] = int(value)
            elif isinstance(value, (np.floating,)):
                decoded[key] = float(value)
            else:
                decoded[key] = value

        logger.info(f"Best Hyperparameters (decoded): {decoded}")

if __name__ == "__main__":
        main()

