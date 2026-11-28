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
from pathlib import Path

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
    real tile images from one dataset split laid out as
    <folder_path>/<class>/<image>. It keeps no random state of its own:
    every random choice uses the rng passed to get_tile, so one source
    can be shared by several environments. Each tile records the file it
    came from in tile.info["source"].

    INPUTS: folder_path, classes, size

    OUTPUTS: None
    """
    def __init__(self, folder_path, classes=None, size=TILE_SIZE):
        """
        FUNCTION NAME: __init__

        DESCRIPTION: This is the constructor for the FolderTileSource class.
        It indexes the image files of every class folder up front.

        INPUTS: folder_path (str or Path, e.g. "Image-Data/test"),
        classes (list of str, or None for the sorted subfolder names),
        size (int, tile width and height in pixels)

        OUTPUTS: None
        """
        folder = Path(folder_path)
        if not folder.is_dir():
            raise FileNotFoundError(f"Tile folder {folder} does not exist.")
        if not isinstance(size, numbers.Integral):
            raise TypeError("The size of the tile must be an integer.")
        if size <= 0:
            raise ValueError("The size of the tile must be a positive integer.")

        # Default to the class subfolders, sorted so the order is stable
        if classes is None:
            classes = sorted(entry.name for entry in folder.iterdir() if entry.is_dir())
        if len(classes) < 2:
            raise ValueError(f"{folder} must contain at least 2 class folders.")
        if len(classes) != len(set(classes)):
            raise ValueError("Duplicate classes are not allowed.")

        # Index every image file per class
        self.files = {}
        for label in classes:
            class_dir = folder / label
            if not class_dir.is_dir():
                raise FileNotFoundError(f"Class folder {class_dir} does not exist.")
            paths = sorted(path for path in class_dir.iterdir()
                           if path.is_file() and path.suffix.lower() in IMAGE_EXT)
            if not paths:
                raise ValueError(f"Class folder {class_dir} has no images ({', '.join(IMAGE_EXT)}).")
            self.files[label] = paths

        self.folder_path = folder
        self.classes = list(classes)
        self.size = size

    def get_tile(self, label, rng):
        """
        FUNCTION NAME: get_tile

        DESCRIPTION: This function picks a random image of the given class
        (using only rng), converts it to RGB, and resizes it to size x size.
        The file path is stored in tile.info["source"].

        INPUTS: label (str), rng (random.Random)

        OUTPUTS: PIL Image object for the tile
        """
        if label not in self.classes:
            raise ValueError(f"label must be one of the following: {self.classes}")
        if not isinstance(rng, random.Random):
            raise TypeError("rng must be an instance of random.Random.")

        path = rng.choice(self.files[label])
        with Image.open(path) as image:
            tile = image.convert("RGB").resize((self.size, self.size))
        tile.info["source"] = str(path)
        return tile


#############################################################
#   FUNCTIONS
#############################################################