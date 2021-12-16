from __future__ import annotations
from typing import Callable, Any, NewType
import inspect
import numpy as np
from enum import Enum, auto
from napari.layers import Shapes
from napari.types import ImageData, LabelsData

Array1D = NewType("Array1D", np.ndarray)

class FilterShapes(Shapes):
    def __init__(self, function: Callable[[np.ndarray], Any]):
        sig = inspect.signature(function)
        
        if len(sig.parameters) != 0:
            raise TypeError("Input args must be 1.")
        
        return_annotation = sig.return_annotation
        if return_annotation is inspect._empty:
            # TODO: dry run
            out = function(np.zeros((2, 2), dtype=np.float32))
        
            if isinstance(out, np.ndarray):
                if out.ndim == 2:
                    return_annotation = ImageData
                elif out.ndim == 1:
                    return_annotation = Array1D
                else:
                    return_annotation = Any
                    
        self.function = function
        self.return_annotation = return_annotation

        super().__init__(ndim=2, face_color="transparent", edge_color="yellow", mode="add_rectangle")