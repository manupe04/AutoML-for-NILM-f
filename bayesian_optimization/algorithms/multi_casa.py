"""Entrenamiento con varias casas y normalizacion configurable.

Mixin para los desagregadores de ventana (seq2point, SGN, TreeCNN, seq2seq).
Cada clase solo define `_ventanas_entrenamiento(mains, aparato)`: como arma
X e Y a partir de dos series ya normalizadas y alineadas.

Normalizacion:
  - "pico" (la original): agregado y aparato divididos por el maximo del
    agregado de train.
  - "fija": agregado estandarizado con la media y el desvio de train, y
    aparato dividido por una potencia tipica del aparato (`potencia_aparato`).

Experimento local en REFIT (30/09/2026, 60 dias de train, 10 epocas): con
una sola casa los modelos aprenden el nivel de consumo de esa casa (la
heladera de la casa 5 consume 64 W de media, la de test 37 W) y predicen
1,4-7,5 veces la energia real. Con 3 casas y normalizacion fija, la heladera
y el microondas bajan su MAE de test (heladera 35-41 -> 30-33 W; microondas
21-34 -> 11-13 W) y la energia predicha queda en 1,05-2,3 veces la real. En el
lavarropas no mejoro, por eso es opcional y lo decide el plan por dataset.
"""
import numpy as np
import pandas as pd

from algorithms.corte_temprano import fit_con_corte

NORMALIZACIONES = ("pico", "fija")


class EntrenamientoMultiCasa:

    BATCH_SIZE = 128  # TreeCNN usa 64, como su train() original
    normalizacion = "pico"
    mu = None
    sd = None
    potencia_aparato = None

    def configurar_normalizacion(self, normalizacion="pico", potencia_aparato=None):
        if normalizacion not in NORMALIZACIONES:
            raise ValueError(f"normalizacion '{normalizacion}': usar {NORMALIZACIONES}")
        if normalizacion == "fija" and not potencia_aparato:
            raise ValueError("la normalizacion fija necesita potencia_aparato")
        self.normalizacion = normalizacion
        self.potencia_aparato = potencia_aparato

    # Normalizacion: x = agregado (entrada), y = aparato (salida)
    def _nx(self, x):
        if self.normalizacion == "fija":
            return (x - self.mu) / self.sd
        return x / self.mmax

    def _ny(self, y):
        if self.normalizacion == "fija":
            return y / self.potencia_aparato
        return y / self.mmax

    def _dy(self, y):
        if self.normalizacion == "fija":
            return y * self.potencia_aparato
        return y * self.mmax

    def train(self, mains, meter, epochs=1, batch_size=None, **load_kwargs):
        self.train_casas([(mains, meter)], epochs=epochs, batch_size=batch_size, **load_kwargs)

    def train_casas(self, pares, epochs=1, batch_size=None, **load_kwargs):
        """Entrena con una lista de (mains, meter), una por casa.

        Las ventanas se arman por casa (no cruzan de una casa a otra), se
        juntan y se entrena una sola vez. El corte temprano valida con el
        ultimo 10% del total, o sea el final del tramo de la ultima casa.
        """
        batch_size = batch_size or self.BATCH_SIZE
        datos = []
        for mains, meter in pares:
            g = pd.concat(list(mains.power_series(**load_kwargs))).fillna(0)
            a = pd.concat(list(meter.power_series(**load_kwargs))).fillna(0)
            ix = g.index.intersection(a.index)
            datos.append((g[ix], a[ix]))
        agregado = pd.concat([g for g, _ in datos])
        if self.mmax is None:
            self.mmax = float(agregado.max())
        if self.normalizacion == "fija" and self.mu is None:
            self.mu, self.sd = float(agregado.mean()), float(agregado.std())

        ventanas = [self._ventanas_entrenamiento(self._nx(g), self._ny(a)) for g, a in datos]
        X = np.concatenate([v[0] for v in ventanas])
        Y = np.concatenate([v[1] for v in ventanas])
        self.stopped_epoch = max(self.stopped_epoch,
                                 fit_con_corte(self.model, X, Y, epochs, batch_size, self.patience))

    # Exportar / importar la normalizacion junto con el modelo
    def _guardar_normalizacion(self, grupo):
        grupo.create_dataset('normalizacion', data=[self.normalizacion.encode()])
        if self.normalizacion == "fija":
            grupo.create_dataset('mu', data=[self.mu])
            grupo.create_dataset('sd', data=[self.sd])
            grupo.create_dataset('potencia_aparato', data=[self.potencia_aparato])

    def _leer_normalizacion(self, grupo):
        norm = grupo.get('normalizacion')
        self.normalizacion = np.array(norm)[0].decode() if norm is not None else "pico"
        if self.normalizacion == "fija":
            self.mu = float(np.array(grupo.get('mu'))[0])
            self.sd = float(np.array(grupo.get('sd'))[0])
            self.potencia_aparato = float(np.array(grupo.get('potencia_aparato'))[0])


def abrir_casas_extra(dataset_path, train_extra, meter_key):
    """Abre las casas de train adicionales: [(building, inicio, fin), ...].

    Devuelve los pares (mains, meter) para train_casas() y los DataSet
    abiertos, para cerrarlos despues de entrenar.
    """
    from nilmtk import DataSet
    pares, abiertos = [], []
    for building, inicio, fin in (train_extra or []):
        ds = DataSet(dataset_path)
        ds.set_window(start=inicio, end=fin)
        elec = ds.buildings[int(building)].elec
        pares.append((elec.mains(), elec.submeters()[meter_key]))
        abiertos.append(ds)
    return pares, abiertos
