#The class ClassTest is a simple example of what a class can look like and do

class ClassTest():
    def __init__(self, layers):
        self._layers = layers

    def setLayers(self, layers):
        self._layers = layers
        
    def getLayers(self):
        return self._layers
