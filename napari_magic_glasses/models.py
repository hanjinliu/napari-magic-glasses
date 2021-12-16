from __future__ import annotations
from typing import Callable, Any, Iterable, NewType
import inspect
import weakref
import numpy as np
import napari
from napari.layers import Image, Shapes
from napari.types import ImageData, LabelsData

Array1D = NewType("Array1D", np.ndarray)

class DelayedArray:
    def __init__(self, data: np.ndarray, function: Callable[[np.ndarray], Any]):
        self.data = data
        self.function = function
    
    def __getitem__(self, sl):
        return self.__class__(self.data[sl], self.function)
    
    def __array__(self, dtype=None):
        return self.function(self.data)
    
    def compute(self):
        return self.function(self.data)

    def transpose(self, axes):
        return self.__class__(self.data.transpose(axes), self.function)
    
    @property
    def shape(self):
        return self.data.shape
    
    @property
    def ndim(self):
        return self.data.ndim
    
    
class FilterShapes(Shapes):
    _MIN_SIZE = 16
    def __init__(self, function: Callable[[np.ndarray], Any], name=None):
        self.function = function
        self._cache = {}
        self.image_anchor = [0, 0]
        self.image_layer_ref: weakref.WeakValueDictionary[int, Image] = weakref.WeakValueDictionary()
        self._image_data = None

        super().__init__(
            ndim=2,
            face_color="transparent",
            edge_color="yellow",
            edge_width=3,
            name=name or function.__name__
            )
        self.mode = "add_rectangle"
        self.events.set_data.connect(lambda e: self.apply_function())
    
    @property
    def function(self) -> Callable[[np.ndarray], Any]:
        return self._function
    
    @function.setter
    def function(self, func):
        sig = inspect.signature(func)
        
        if len(sig.parameters) < 1:
            raise TypeError("Input args must be >=1.")
        
        return_annotation = sig.return_annotation
        if return_annotation is inspect._empty:
            # dry run
            out = func(np.zeros((2, 2), dtype=np.float32))
        
            if isinstance(out, np.ndarray):
                if out.ndim == 2:
                    return_annotation = ImageData
                elif out.ndim == 1:
                    return_annotation = Array1D
                else:
                    return_annotation = Any
                    
        self._function = func
        self._return_annotation = return_annotation
    
    @property
    def image_anchor(self) -> np.ndarray:
        return self._image_anchor
    
    @image_anchor.setter
    def image_anchor(self, shift: Iterable[float] | str):
        if isinstance(shift, str):
            raise NotImplementedError()
        else:
            dy, dx = shift
        self._image_anchor = np.array([dy, dx], dtype=np.float64)
    
    @property
    def image_data(self) -> DelayedArray:
        return self._image_data
    
    @image_data.setter
    def image_data(self, data: Any):
        if isinstance(data, np.ndarray):
            self._image_data = DelayedArray(data, self.function)
            self._image_scale = np.ones(2)
        elif isinstance(data, Image):
            self._image_data = DelayedArray(data.data, self.function)
            self._image_scale = data.scale[-2:]
        else:
            raise TypeError(f"{type(data)}")
            
    
    def apply_function(self):
        # Extract image layer data at the rectangle coordinates and apply function.
        if self._image_data is None:
            viewer = napari.current_viewer()
            img = front_image(viewer)
            if img is None:
                return
            self.image_data = img
            
        for i in range(self.nshapes):
            if self.shape_type[i] != "rectangle":
                continue
        
            rect = self.data[i] / self._image_scale
            coords = normalize_rectangle(rect, self._image_data.shape[-2:])
            y0, y1, x0, x1 = coords
            
            if y1 - y0 <= self._MIN_SIZE or x1 - x0 <= self._MIN_SIZE:
                continue
            
            # d = tuple(viewer.dims.current_step[:-2])
            sl = (..., slice(y0, y1), slice(x0, x1))

            out = self._image_data[sl]
            translate = self.image_anchor + np.array([y0, x0])
            if i in self.image_layer_ref.keys():
                image_layer = self.image_layer_ref[i]
                image_layer.data = out
                image_layer.translate = translate
                image_layer.scale = self._image_scale
                
            else:
                image_layer = Image(out, translate=translate, scale=self._image_scale)
                self.image_layer_ref[i] = image_layer
                layers = napari.current_viewer().layers
                layers.insert(0, image_layer)
                

def front_image(viewer: "napari.Viewer"):
    """
    From list of image layers return the most front visible image.
    """        
    front = None
    
    for layer in reversed(viewer.layers):
        if isinstance(layer, Image) and layer.visible:
            front = layer
            break
    
    return front

def normalize_rectangle(rect: np.ndarray, image_shape: tuple[int, int]) -> tuple[int, int, int, int]:
    y0, x0 = np.min(rect, axis=0).astype(np.int64)
    y1, x1 = np.ceil(np.max(rect, axis=0) + 1).astype(np.int64)
    sizey, sizex = image_shape
    y0 = max(0, y0)
    y1 = min(y1, sizey)
    x0 = max(0, x0)
    x1 = min(x1, sizex)
    return y0, y1, x0, x1