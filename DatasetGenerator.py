import os
from dataclasses import dataclass
from random import choice, randint
from typing import List

tipos = {
    'camisa': 0,
    'pantalon': 1,
    'buso': 2
}

tallas = {
    'XS': 0,
    'S': 1,
    'M': 2,
    'L': 3,
    'XL': 4,
}

colores = {
    'rojo': 0,
    'azul': 1,
    'verde': 2,
    'fucsia': 3,
}

class EntradaRopa:
    def __init__(self, tipo: int, talla: int, color: int, ventas: int):
        self.tipo = tipo
        self.talla = talla
        self.color = color
        self.ventas = ventas

    def __repr__(self):
        return f'{self.tipo},{self.talla},{self.color},{self.ventas}\n'


class DatasetGenerator:
    def __call__(self, data_size=1000):
        data = []
        tipo_opciones = list(tipos.values())
        talla_opciones = list(tallas.values())
        color_opciones = list(colores.values())

        for i in range(data_size):
            tipo = choice(tipo_opciones)
            talla = choice(talla_opciones)
            color = choice(color_opciones)
            upper_limit = 100 + (tipo * talla * color * 100)
            data.append(
                EntradaRopa(
                    tipo=tipo,
                    talla=talla,
                    color=color,
                    ventas=randint(10, upper_limit),
                )
            )
        self.__save_data(data)

    def __save_data(self, data: List[EntradaRopa]):
        if os.path.exists('./dataset.csv'):
            return
        with open('dataset.csv', 'w') as f:
            f.writelines(['tipo,talla,color,ventas\n'])
            f.writelines([repr(i) for i in data])



