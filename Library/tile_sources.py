#############################################################
#   FILE NAME: tile_sources.py
#
#   DESCRIPTION: This is the tile sources file in which
#   different types of tile sources are defined.
#
#   Authors: Group 2
#############################################################

#############################################################
#   Libraries
#############################################################

import numbers
import colorsys
import random

from PIL import Image, ImageDraw

#############################################################
#   Files Import
#############################################################

#############################################################
#   Global Constants
#############################################################

TILE_SIZE = 100

IMAGE_EXT = [".jpg", ".jpeg", ".png"]

DEFAULT_CLASSES = [
    "bus", 
    "bicycle", 
    "traffic_light", 
    "hydrant"
    ]

#############################################################
#   CLASSES
#############################################################

class SyntheticTileSource:
    """
    CLASS NAME: SyntheticTileSource

    DESCRIPTION: This class represents a synthetic tile source, which generates
    tile images programmatically.

    INPUTS: classes, size, label, rng

    OUTPUTS: None
    """
    def __init__(self, classes=None, size=TILE_SIZE):
        """
        FUNCTION NAME: __init__

        DESCRIPTION: This is the constructor for the SyntheticTileSource class.

        INPUTS: classes, size

        OUTPUTS: None
        """
        # Fill in the provided classes and size if they are provided
        if classes is None:
            classes = DEFAULT_CLASSES
        # Atleast 2 classes are required to generate synthetic tiles
        if len(classes) < 2:
            raise ValueError("The number of classes must be at least 2.")

        # No duplicate classes are allowed
        if len(classes) != len(set(classes)):
            raise ValueError("Duplicate classes are not allowed.")
        # The size of the tile must be a positive integer
        if not isinstance(size, numbers.Integral):
            raise TypeError("The size of the tile must be an integer.")
        if size <= 0:
            raise ValueError("The size of the tile must be a positive integer.")
        # Set self variables
        self.classes = list(classes)
        self.size = size
        self.colors = {}
        for index, word in enumerate(self.classes):
            hue = index / len(self.classes)
            rgb = colorsys.hsv_to_rgb(hue, 0.4, 0.4)
            self.colors[word] = tuple(int(c * 255) for c in rgb)

    def get_tile(self, label, rng):
        """
        FUNCTION NAME: get_tile
        
        DESCRIPTION: This function generates a tile image for a given class from the specified tile source.

        INPUTS: tile_source, tile_class, size

        OUTPUTS: PIL Image object representing the generated tile
        """
        if label not in self.classes:
            raise ValueError(f"label must be one of the following: {self.classes}")
        if not isinstance(rng, random.Random):
            raise TypeError("rng must be an instance of random.Random.")

        tile = Image.new("RGB", (self.size, self.size), self.colors[label])
        ImageDraw.Draw(tile).text((10, 10), label.replace("_", " "), fill=(255, 255, 255))
        return tile


class FolderTileSource:
    """
    CLASS NAME: FolderTileSource

    DESCRIPTION: This class represents a folder tile source, which loads
    tile images from a specified folder.

    INPUTS: folder_path, classes, size

    OUTPUTS: None
    """
    def __init__(self, folder_path, classes=None, size=TILE_SIZE):
        """
        FUNCTION NAME: __init__

        DESCRIPTION: This is the constructor for the FolderTileSource class.

        INPUTS: folder_path, classes, size

        OUTPUTS: None
        """
        self.folder_path = folder_path
        if classes is None:
            classes = DEFAULT_CLASSES
        self.classes = list(classes)
        self.size = size
        self.tiles = {}

        
#############################################################
#   FUNCTIONS
#############################################################