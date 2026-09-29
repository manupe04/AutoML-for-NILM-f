from __future__ import print_function, division

import numpy as np
import pandas as pd
import h5py
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import (Conv1D, Conv1DTranspose, BatchNormalization,
                                     Input, ReLU)
from nilmtk.legacy.disaggregate import Disaggregator


def _make_block(window_size, name):
    """Bloque convolucional encoder-decoder para un electrodomestico.

    Port a 1D del CustomCNN del repositorio oficial de los autores
    (yilingjia/TreeCNN-for-Energy-Breakdown), que en el original usa Conv2D
    sobre una representacion dias x horas. Aca se usa Conv1D porque el
    pipeline de NILMTK entrega ventanas unidimensionales.

    Se mantiene la estructura del original: dos convoluciones que comprimen
    y dos transpuestas que reconstruyen, con BatchNorm entre medio. La salida
    tiene la misma longitud que la entrada (seq2seq, no seq2point).
    """
    inp = Input(shape=(window_size, 1), name=name + "_in")

    x = Conv1D(16, 7, padding="same", name=name + "_conv1")(inp)
    x = ReLU(name=name + "_act1")(x)
    x = BatchNormalization(name=name + "_bn1")(x)

    x = Conv1D(32, 2, strides=2, padding="same", name=name + "_conv2")(x)
    x = BatchNormalization(name=name + "_bn2")(x)

    x = Conv1DTranspose(16, 2, strides=2, padding="same", name=name + "_deconv1")(x)
    x = BatchNormalization(name=name + "_bn3")(x)

    out = Conv1DTranspose(1, 7, padding="same", name=name + "_deconv2")(x)

    return Model(inp, out, name=name)


class TreeCNNCascade(tf.keras.Model):
    """Cascada de bloques CNN con resta progresiva del agregado.

    Es el nucleo de TreeCNN. Para cada electrodomestico, en el orden
    configurado:

        1. El bloque predice el consumo del aparato a partir del agregado
           que queda disponible en ese punto de la cascada.
        2. Esa prediccion se resta del agregado.
        3. El bloque siguiente trabaja sobre el residuo.

    Durante el entrenamiento se aplica scheduled sampling: con probabilidad
    `teacher_forcing_prob` se resta el consumo real (ground truth) en lugar
    de la prediccion. Eso evita que los errores de los primeros bloques se
    propaguen y desestabilicen el entrenamiento de los siguientes.

    En inferencia siempre se resta la prediccion, porque el ground truth
    no esta disponible.
    """

    def __init__(self, window_size, appliance_order, teacher_forcing_prob=0.5, **kwargs):
        super().__init__(**kwargs)
        self.window_size = window_size
        self.appliance_order = list(appliance_order)
        self.teacher_forcing_prob = teacher_forcing_prob
        self.blocks = [
            _make_block(window_size, "block_" + str(i) + "_" + a.replace(" ", "_"))
            for i, a in enumerate(self.appliance_order)
        ]

    def call(self, inputs, training=False):
        agg = inputs
        preds = []
        for block in self.blocks:
            p = block(agg, training=training)
            preds.append(p)
            agg = agg - p
        return tf.concat(preds, axis=-1)

    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            agg = x
            preds = []
            for i, block in enumerate(self.blocks):
                p = block(agg, training=True)
                preds.append(p)
                gt_i = y[..., i:i + 1]
                # Scheduled sampling: resta la prediccion o el valor real.
                use_pred = tf.random.uniform([]) > self.teacher_forcing_prob
                agg = agg - tf.cond(use_pred, lambda: p, lambda: gt_i)
            out = tf.concat(preds, axis=-1)
            # self.compiled_loss / self.compiled_metrics quedaron deprecados en
            # Keras 3 (el que trae TF 2.19 del repo); con ellos el loss subia en
            # vez de bajar porque el shim viejo no calculaba bien el gradiente.
            loss = self.compute_loss(x=x, y=y, y_pred=out)

        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        return self.compute_metrics(x, y, out)


class TreeCNNDisaggregator(Disaggregator):
    """TreeCNN (Jia et al., 2019) adaptado a la interfaz de AutoML4NILM.

    IMPORTANTE - desajuste de interfaz:

    TreeCNN es un modelo MULTI-electrodomestico por diseno: su aporte esta
    en descomponer el agregado en cascada, restando cada aparato antes de
    estimar el siguiente. La interfaz de AutoML4NILM, en cambio, entrena un
    aparato por vez (`meter_key=appliance`).

    Por eso esta clase expone dos caminos:

      - `train_multi(mains, meters)`: el TreeCNN real. Recibe un diccionario
        de medidores y entrena la cascada completa. Es el que hay que usar
        para aprovechar el modelo.

      - `train(mains, meter)`: el camino compatible con el pipeline. Con un
        solo medidor la cascada degenera en un unico bloque encoder-decoder.
        Sirve para que el modelo entre en la optimizacion bayesiana y sea
        comparable con los demas, pero NO explota la estructura en cascada.

    El modelo esta pensado para datos de BAJA FRECUENCIA (una muestra cada
    15 minutos o por hora), que es el regimen de medicion disponible en las
    cooperativas electricas.
    """

    def __init__(self, patience, optimizer, learning_rate, loss,
                 window_size=128, appliance_order=None, teacher_forcing_prob=0.5):
        self.MODEL_NAME = "TreeCNN"
        self.mmax = None
        self.patience = patience
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.loss = loss
        # La ventana debe ser par: hay una convolucion con stride 2 y su
        # transpuesta correspondiente.
        if window_size % 2 != 0:
            window_size += 1
        self.window_size = window_size
        self.appliance_order = appliance_order or ["target"]
        self.teacher_forcing_prob = teacher_forcing_prob
        self.target_index = 0
        self.stopped_epoch = 0
        self.model = self._create_model(self.optimizer, self.learning_rate, self.loss)

    def _create_model(self, optimizer, learning_rate, loss):
        model = TreeCNNCascade(
            window_size=self.window_size,
            appliance_order=self.appliance_order,
            teacher_forcing_prob=self.teacher_forcing_prob,
        )
        model.build((None, self.window_size, 1))
        model.compile(optimizer=optimizer, loss=loss)
        return model

    def _normalize(self, chunk, mmax):
        return chunk / mmax

    def _denormalize(self, chunk, mmax):
        return chunk * mmax

    def _create_windows(self, series):
        """Ventanas NO solapadas, porque la salida es seq2seq.

        Seq2point predice un punto por ventana y por eso desliza de a uno.
        Aca cada ventana produce una ventana completa, asi que se recorre
        sin solapamiento y despues se concatena.
        """
        values = np.asarray(series, dtype="float32")
        n = len(values)
        n_win = int(np.ceil(n / self.window_size))
        padded = np.pad(values, (0, n_win * self.window_size - n), mode="constant")
        X = padded.reshape(n_win, self.window_size, 1)
        return X, pd.Index(series.index)

    # ------------------------------------------------------------------
    # Camino multi-electrodomestico: el TreeCNN real
    # ------------------------------------------------------------------
    def train_multi(self, mains, meters, epochs=1, batch_size=64, **load_kwargs):
        """Entrena la cascada completa.

        `meters` es un diccionario {nombre_electrodomestico: ElecMeter}.
        El orden de las claves define el orden de la cascada, que en el
        paper importa: conviene poner primero los aparatos de mayor
        consumo, porque son los que mas limpian el residuo.
        """
        self.appliance_order = list(meters.keys())
        self.model = self._create_model(self.optimizer, self.learning_rate, self.loss)

        main_series = mains.power_series(**load_kwargs)
        meter_series = {k: m.power_series(**load_kwargs) for k, m in meters.items()}

        mainchunk = next(main_series)
        meterchunks = {k: next(v) for k, v in meter_series.items()}

        if self.mmax is None:
            self.mmax = mainchunk.max()

        run = True
        while run:
            mainchunk = self._normalize(mainchunk, self.mmax)
            meterchunks = {k: self._normalize(v, self.mmax) for k, v in meterchunks.items()}
            self.train_on_chunk_multi(mainchunk, meterchunks, epochs, batch_size)
            try:
                mainchunk = next(main_series)
                meterchunks = {k: next(v) for k, v in meter_series.items()}
            except:
                run = False

    def train_on_chunk_multi(self, mainchunk, meterchunks, epochs, batch_size):
        mainchunk = mainchunk.fillna(0)
        meterchunks = {k: v.fillna(0) for k, v in meterchunks.items()}

        ix = mainchunk.index
        for v in meterchunks.values():
            ix = ix.intersection(v.index)
        mainchunk = mainchunk[ix]
        meterchunks = {k: v[ix] for k, v in meterchunks.items()}

        X, _ = self._create_windows(mainchunk)

        ys = []
        for name in self.appliance_order:
            Y, _ = self._create_windows(meterchunks[name])
            ys.append(Y)
        Y = np.concatenate(ys, axis=-1)

        self.model.fit(X, Y, epochs=epochs, batch_size=batch_size, shuffle=True)

    # ------------------------------------------------------------------
    # Camino compatible con AutoML4NILM: un solo electrodomestico
    # ------------------------------------------------------------------
    def train(self, mains, meter, epochs=1, batch_size=64, **load_kwargs):
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
        mainchunk = mainchunk.fillna(0)
        meterchunk = meterchunk.fillna(0)

        ix = mainchunk.index.intersection(meterchunk.index)
        mainchunk = mainchunk[ix]
        meterchunk = meterchunk[ix]

        X, _ = self._create_windows(mainchunk)
        Y, _ = self._create_windows(meterchunk)

        self.model.fit(X, Y, epochs=epochs, batch_size=batch_size, shuffle=True)

    def disaggregate_chunk(self, mains):
        mains = mains.fillna(0)
        normalized = self._normalize(mains, self.mmax)
        X, index = self._create_windows(normalized)

        preds = self.model.predict(X, batch_size=64)
        # Se toma la rama del electrodomestico objetivo y se reconstruye
        # la serie concatenando las ventanas.
        target = preds[..., self.target_index].reshape(-1)
        target = target[:len(index)]
        target = self._denormalize(target, self.mmax)

        return pd.DataFrame({0: target}, index=index)

    def disaggregate(self, mains, output_datastore, meter_metadata, **load_kwargs):
        load_kwargs = self._pre_disaggregation_checks(load_kwargs)
        load_kwargs.setdefault('sample_period', 60)
        load_kwargs.setdefault('sections', mains.good_sections())

        timeframes = []
        building_path = f'/building{mains.building()}'
        mains_data_location = building_path + '/elec/meter1'
        data_is_available = False

        for chunk in mains.power_series(**load_kwargs):
            if len(chunk) < 100:
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
        # compile=False: Keras 3 no puede deserializar la loss compilada
        # ('mse') guardada en el formato legacy H5 (ver misma nota en
        # sgndisaggregator.py). Ademas TreeCNNCascade es una subclase de
        # tf.keras.Model, asi que hay que declararla via custom_objects para
        # que load_model() sepa reconstruirla.
        self.model = load_model(filename, compile=False,
                                 custom_objects={"TreeCNNCascade": TreeCNNCascade})
        with h5py.File(filename, 'r') as hf:
            self.mmax = np.array(hf.get('disaggregator-data').get('mmax'))[0]
            self.window_size = int(np.array(hf.get('disaggregator-data').get('window_size'))[0])

    def export_model(self, filename):
        self.model.save(filename)
        with h5py.File(filename, 'a') as hf:
            gr = hf.create_group('disaggregator-data')
            gr.create_dataset('mmax', data=[self.mmax])
            gr.create_dataset('window_size', data=[self.window_size])
