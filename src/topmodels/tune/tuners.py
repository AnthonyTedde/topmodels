from abc import ABC, abstractmethod


class TunerBase(ABC):
    @abstractmethod
    def suggest_int(self):
        pass
    @abstractmethod
    def suggest_float(self):
        pass

    @abstractmethod
    def suggest_categorical(self):
        pass


class Tuner(TunerBase):
    pass