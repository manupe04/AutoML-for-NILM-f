"""Sequence-to-sequence (Zhang et al., 2018, "Sequence-to-point learning with
neural networks for non-intrusive load monitoring", AAAI).

Cada ventana de W muestras del agregado predice la ventana completa de W
muestras del aparato (seq2point predice solo el punto central). La red es la
misma CNN del paper para ambos: 5 convoluciones, una densa de 1024 y la salida.

En inferencia las ventanas se deslizan de a una muestra, asi que cada instante
queda cubierto por hasta W predicciones: la salida es su promedio.

La version anterior de este archivo era una copia de WindowGRU (misma red,
una sola salida): daba resultados identicos a window gru.
"""
from __future__ import print_function, division

import numpy as np
import pandas as pd
import h5py
from numpy.lib.stride_tricks import sliding_window_view
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Input, Conv1D, Dense, Flatten
from nilmtk.legacy.disaggregate import Disaggregator
from algorithms.multi_casa import EntrenamientoMultiCasa


class Seq2SeqDisaggregator(EntrenamientoMultiCasa, Disaggregator):
    def __init__(self, patience, optimizer, learning_rate, loss, window_size=99):
        self.MODEL_NAME = "Seq2Seq"
        self.mmax = None
        self.patience = patience
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.loss = loss
        self.window_size = window_size
        self.MIN_CHUNK_LENGTH = 100
        self.stopped_epoch = 0
        self.model = self._create_model(self.optimizer, self.learning_rate, self.loss)

    def _create_model(self, optimizer, learning_rate, loss):
        w = self.window_size
        model = Sequential([
            Input(shape=(w, 1)),
            Conv1D(30, 10, activation='relu', padding='same'),
            Conv1D(30, 8, activation='relu', padding='same'),
            Conv1D(40, 6, activation='relu', padding='same'),
            Conv1D(50, 5, activation='relu', padding='same'),
            Conv1D(50, 5, activation='relu', padding='same'),
            Flatten(),
            Dense(1024, activation='relu'),
            Dense(w, activation='linear'),
        ])
        model.compile(optimizer=optimizer, loss=loss)
        return model

    def _normalize(self, chunk, mmax):
        return chunk / mmax

    def _denormalize(self, chunk, mmax):
        return chunk * mmax

    def _windows(self, values):
        """Ventanas de W muestras con paso 1 (n - W + 1 ventanas).

        Si la serie es mas corta que W se completa con ceros al final.
        """
        values = np.asarray(values, dtype='float32')
        if len(values) < self.window_size:
            values = np.pad(values, (0, self.window_size - len(values)))
        return sliding_window_view(values, self.window_size)

    def _ventanas_entrenamiento(self, mains, aparato):
        # train() y train_casas() vienen de EntrenamientoMultiCasa
        return self._windows(mains)[..., np.newaxis], self._windows(aparato)

    def disaggregate_chunk(self, mains):
        mains = mains.fillna(0)
        n = len(mains)
        w = self.window_size
        X = self._windows(self._nx(mains))
        pred = self.model.predict(X[..., np.newaxis], batch_size=128)

        # Promedio de las predicciones solapadas: la ventana i cubre i..i+W-1
        total = np.zeros(max(n, w), dtype='float64')
        cuenta = np.zeros(max(n, w), dtype='float64')
        k = len(pred)
        for j in range(w):
            total[j:j + k] += pred[:, j]
            cuenta[j:j + k] += 1
        promedio = (total / cuenta)[:n]

        predictions = self._dy(promedio)
        return pd.DataFrame({0: predictions}, index=mains.index)

    def disaggregate(self, mains, output_datastore, meter_metadata, **load_kwargs):
        load_kwargs = self._pre_disaggregation_checks(load_kwargs)
        load_kwargs.setdefault('sample_period', 60)
        load_kwargs.setdefault('sections', mains.good_sections())

        timeframes = []
        building_path = f'/building{mains.building()}'
        mains_data_location = building_path + '/elec/meter1'
        data_is_available = False

        for chunk in mains.power_series(**load_kwargs):
            if len(chunk) < self.MIN_CHUNK_LENGTH:
                continue
            print("New sensible chunk: {}".format(len(chunk)))

            timeframes.append(chunk.timeframe)
            measurement = chunk.name

            appliance_power = self.disaggregate_chunk(chunk)
            appliance_power[appliance_power < 0] = 0

            data_is_available = True
            cols = pd.MultiIndex.from_tuples([chunk.name])
            meter_instance = meter_metadata.instance()
            df = pd.DataFrame(appliance_power.values, index=appliance_power.index, columns=cols, dtype="float32")
            key = f'{building_path}/elec/meter{meter_instance}'
            output_datastore.append(key, df)

            mains_df = pd.DataFrame(chunk, columns=cols, dtype="float32")
            output_datastore.append(key=mains_data_location, value=mains_df)

        if data_is_available:
            self._save_metadata_for_disaggregation(
                output_datastore=output_datastore,
                sample_period=load_kwargs['sample_period'],
                measurement=measurement,
                timeframes=timeframes,
                building=mains.building(),
                meters=[meter_metadata]
            )

    def import_model(self, filename):
        self.model = load_model(filename, compile=False)
        with h5py.File(filename, 'r') as hf:
            self.mmax = np.array(hf.get('disaggregator-data').get('mmax'))[0]
            self.window_size = int(np.array(hf.get('disaggregator-data').get('window_size'))[0])
            self._leer_normalizacion(hf.get('disaggregator-data'))

    def export_model(self, filename):
        self.model.save(filename)
        with h5py.File(filename, 'a') as hf:
            gr = hf.create_group('disaggregator-data')
            gr.create_dataset('mmax', data=[self.mmax])
            gr.create_dataset('window_size', data=[self.window_size])
            self._guardar_normalizacion(gr)
