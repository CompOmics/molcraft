import keras

from molcraft import chem


@keras.saving.register_keras_serializable(package="molcraft")
class ConformerProcessor:

    def get_config(self) -> dict:
        return {}

    @classmethod
    def from_config(cls, config: dict):
        return cls(**config)

    def __call__(self, mol: chem.Mol) -> chem.Mol:
        raise NotImplementedError


@keras.saving.register_keras_serializable(package="molcraft")
class ConformerEmbedder(ConformerProcessor):

    def __init__(
        self, 
        method: str = 'ETKDGv3',
        num_conformers: int = 5, 
        **kwargs,
    ) -> None:
        self.method = method 
        self.num_conformers = num_conformers 
        self.kwargs = kwargs 

    def get_config(self) -> dict:
        config = {
            'method': self.method, 
            'num_conformers': self.num_conformers, 
        }
        config.update({
            k: v for (k, v) in self.kwargs.items()
        })
        return config
    
    def __call__(self, mol: chem.Mol) -> chem.Mol:
        return chem.embed_conformers(
            mol, 
            method=self.method,
            num_conformers=self.num_conformers,
            **self.kwargs,
        )


@keras.saving.register_keras_serializable(package="molcraft")
class ConformerOptimizer(ConformerProcessor):

    def __init__(
        self, 
        method: str = 'UFF',
        max_iter: int = 200, 
        ignore_interfragment_interactions: bool = True,
        vdw_threshold: float = 10.0,
        **kwargs,
    ) -> None:
        self.method = method 
        self.max_iter = max_iter 
        self.ignore_interfragment_interactions = ignore_interfragment_interactions
        self.vdw_threshold = vdw_threshold 
        self.kwargs = kwargs

    def get_config(self) -> dict:
        config = {
            'method': self.method,
            'max_iter': self.max_iter,
            'ignore_interfragment_interactions': self.ignore_interfragment_interactions,
            'vdw_threshold': self.vdw_threshold,
        }
        config.update({
            k: v for (k, v) in self.kwargs.items()
        })
        return config
    
    def __call__(self, mol: chem.Mol) -> chem.Mol:
        return chem.optimize_conformers(
            mol,
            method=self.method,
            max_iter=self.max_iter,
            ignore_interfragment_interactions=self.ignore_interfragment_interactions,
            vdw_threshold=self.vdw_threshold,
            **self.kwargs,
        )


@keras.saving.register_keras_serializable(package="molcraft")
class ConformerPruner(ConformerProcessor):
    def __init__(
        self,
        keep: int = 1,
        threshold: float = 0.0,
        energy_force_field: str = 'UFF',
        **kwargs,
    ) -> None:
        self.keep = keep
        self.threshold = threshold
        self.energy_force_field = energy_force_field
        self.kwargs = kwargs

    def get_config(self) -> dict:
        config = {
            'keep': self.keep,
            'threshold': self.threshold,
            'energy_force_field': self.energy_force_field,
        }
        config.update({
            k: v for (k, v) in self.kwargs.items()
        })
        return config
    
    def __call__(self, mol: chem.Mol) -> chem.Mol:
        return chem.prune_conformers(
            mol,
            keep=self.keep,
            threshold=self.threshold,
            energy_force_field=self.energy_force_field,
            **self.kwargs,
        )


@keras.saving.register_keras_serializable(package='molcraft')
class ConformerGenerator(ConformerProcessor):

    def __init__(self, steps: list[ConformerProcessor]) -> None:
        self.steps = steps

    def get_config(self) -> dict:
        return {
            "steps": [
                keras.saving.serialize_keras_object(step) for step in self.steps
            ]
        }

    @classmethod
    def from_config(cls, config: dict) -> 'ConformerGenerator':
        steps = [
            keras.saving.deserialize_keras_object(obj) 
            for obj in config["steps"]
        ]
        return cls(steps)

    def __call__(self, mol: chem.Mol) -> chem.Mol:
        for step in self.steps:
            mol = step(mol)
        return mol