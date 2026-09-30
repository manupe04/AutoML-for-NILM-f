from __future__ import print_function, division
import random
import sys
import pandas as pd
import numpy as np
import h5py
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Dense, Conv1D, GRU, Bidirectional
from nilmtk.legacy.disaggregate import Disaggregator
from algorithms.corte_temprano import fit_con_corte

class WindowGRUDisaggregator(Disaggregator):
    def __init__(self, patience, optimizer, learning_rate, loss, window_size=30):
        self.MODEL_NAME = "WindowGRU"
        self.mmax = None
        self.MIN_CHUNK_LENGTH = 100
        self.patience = patience
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.loss = loss
        self.window_size = window_size
        self.stopped_epoch = 0
        self.model = self._create_model(self.optimizer, self.learning_rate, self.loss)


    def _create_model(self, optimizer, learning_rate, loss):
        model = Sequential()
        model.add(Conv1D(16, 4, activation="relu", padding="same", strides=1, input_shape=(self.window_size, 1)))
        model.add(Conv1D(8, 4, activation="relu", padding="same", strides=1))
        model.add(Bidirectional(GRU(64, return_sequences=True), merge_mode='concat'))
        model.add(Bidirectional(GRU(128, return_sequences=False), merge_mode='concat'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(1, activation='linear'))
        model.compile(loss=loss, optimizer=optimizer)
        return model

    def _normalize(self, chunk, mmax):
        return chunk / mmax

    def _denormalize(self, chunk, mmax):
        return chunk * mmax

    def _create_windows(self, series, window_size):
        values = np.array(series).reshape(-1, 1)
        X, index = [], []
        for i in range(len(values) - window_size + 1):
            X.append(values[i:i + window_size])
            index.append(series.index[i + window_size - 1])
        return np.array(X), pd.Index(index)

    def train(self, mains, meter, epochs=1, batch_size=128, **load_kwargs):
        main_series = mains.power_series(**load_kwargs)
        meter_series = meter.power_series(**load_kwargs)

        run = True
        mainchunk = next(main_series)
        meterchunk = next(meter_series)
        if self.mmax is None:
            self.mmax = mainchunk.max()

        while run:
            mainchunk = self._normalize(mainchunk, self.mmax)
            meterchunk = self._normalize(meterchunk, self.mmax)
            self.train_on_chunk(mainchunk, meterchunk, epochs, batch_size)
            try:
                mainchunk = next(main_series)
                meterchunk = next(meter_series)
            except:
                run = False

    def train_on_chunk(self, mainchunk, meterchunk, epochs, batch_size):
        mainchunk.fillna(0, inplace=True)
        meterchunk.fillna(0, inplace=True)

        ix = mainchunk.index.intersection(meterchunk.index)
        mainchunk = mainchunk[ix]
        meterchunk = meterchunk[ix]

        X, idx = self._create_windows(mainchunk, self.window_size)
        Y = np.array(meterchunk)[self.window_size - 1:]

        self.stopped_epoch = max(self.stopped_epoch, fit_con_corte(self.model, X, Y, epochs, batch_size, self.patience))

    def disaggregate_chunk(self, mains):
        mains.fillna(0, inplace=True)
        X, index = self._create_windows(self._normalize(mains, self.mmax), self.window_size)

        predictions = self.model.predict(X, batch_size=128)
        predictions = self._denormalize(predictions.flatten(), self.mmax)

        return pd.DataFrame({0: predictions}, index=index)

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
        self.model = load_model(filename)
        with h5py.File(filename, 'r') as hf:
            self.mmax = np.array(hf.get('disaggregator-data').get('mmax'))[0]
            self.window_size = int(np.array(hf.get('disaggregator-data').get('window_size'))[0])

    def export_model(self, filename):
        self.model.save(filename)
        with h5py.File(filename, 'a') as hf:
            gr = hf.create_group('disaggregator-data')
            gr.create_dataset('mmax', data=[self.mmax])
            gr.create_dataset('window_size', data=[self.window_size])
