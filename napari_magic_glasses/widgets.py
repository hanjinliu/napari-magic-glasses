from magicclass import magicclass, MagicTemplate, field
from magicgui.widgets import ComboBox
import napari
from napari.layers import Image
from scipy import ndimage as ndi

_FilterLibrary = {
    "Gaussian filter": ndi.gaussian_filter,
    "Sobel filter": ndi.sobel,
}

@magicclass
class MagicGlasses(MagicTemplate):
    image = field(Image)
    function = field(widget_type=ComboBox, options={"choices": list(_FilterLibrary.keys())})
    
    def add_shape(self):
        self.shapes = self.parent_viewer.add_shapes(ndim=2)
        self.shapes.events.data.connect
    
    @function.connect
    def _apply_filter(self):
        self.func = _FilterLibrary[self.function.value]
        