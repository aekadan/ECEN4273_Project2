#############################################################
#   FILE NAME: gui.py
#
#   DESCRIPTION: This file contains the GUI class,
#   which is responsible for creating and managing the
#   graphical user interface of the application.
#
#   Authors: Group 2
#############################################################

#############################################################
#   Libraries
#############################################################
import tkinter as tk

from PIL import ImageTk

#############################################################
#   Files Import
#############################################################

from .captcha_env import CaptchaEnv, GRID_ROWS, GRID_COLS
from .tile_sources import TILE_SIZE

#############################################################
#   Global Constants
#############################################################

SELECTED_COLOR = "#1a73e8"
BORDER_WIDTH = 4

#############################################################
#   CLASSES
#############################################################
class CaptchaApp:
    """
    CLASS NAME: CaptchaApp

    DESCRIPTION: This class is responsible for creating and managing the
    graphical user interface of the application. It initializes the main window,
    and its components.

    INPUTS: env (CaptchaEnv), tile_size (int)

    OUTPUTS: None
    """
    def __init__(self, env: CaptchaEnv, tile_size: int = TILE_SIZE):
        """
        FUNCTION NAME: __init__

        DESCRIPTION: This function initializes the CaptchaApp class, creating the main window
        and its components.

        INPUTS: env (CaptchaEnv), tile_size (int)

        OUTPUTS: None
        """
        # Initialize the class attributes
        self.env = env
        self.tile_size = tile_size
        self.challenge = None
        self.selected_tiles = set()
        self.submitted = False
        self.photo_image_refs = []  # To keep references to PhotoImage objects
        self.score_attempts = 0
        self.score_correct = 0
        self.root = tk.Tk()
        self.unselected_color = self.root.cget("bg")

        # Create the GUI components
        self.root.title("Captcha Challenge")
        self.prompt_label = tk.Label(self.root, text="Select the tiles that match the target:")
        self.prompt_label.pack(pady=5)
        self.tile_frame = tk.Frame(self.root)
        self.tile_frame.pack()

        # Add buttons for submitting and loading the next challenge
        self.controls_frame = tk.Frame(self.root)
        self.controls_frame.pack(pady=5)
        tk.Button(self.controls_frame, text="Submit", command=self.submit).pack(side="left", padx=5)
        tk.Button(self.controls_frame, text="Next", command=self.load_challenge).pack(side="left", padx=5)

        # Add a label for showing the result and score
        self.result_label = tk.Label(self.root, text="Score: 0/0")
        self.result_label.pack(pady=5)

        # Create the 3x3 grid of tiles, stored in index order (0 = top-left)
        self.tile_labels = []
        for row in range(GRID_ROWS):
            for col in range(GRID_COLS):
                tile_index = row * GRID_COLS + col
                tile_label = tk.Label(self.tile_frame, highlightthickness=BORDER_WIDTH,
                                      highlightbackground=self.unselected_color)
                tile_label.grid(row=row, column=col, padx=2, pady=2)
                tile_label.bind("<Button-1>", lambda e, idx=tile_index: self.toggle_tile_selection(idx))
                self.tile_labels.append(tile_label)

        self.load_challenge()


    def load_challenge(self):
        """
        FUNCTION NAME: load_challenge

        DESCRIPTION: This function loads a new challenge from the environment and updates the GUI accordingly.

        INPUTS: None

        OUTPUTS: None
        """
        self.challenge = self.env.new_challenge()
        self.selected_tiles.clear()
        self.submitted = False

        # Convert each tile to a PhotoImage and keep a reference so it is not garbage collected
        self.photo_image_refs = []
        for tile_label, tile in zip(self.tile_labels, self.challenge.tiles):
            resized_tile = tile.resize((self.tile_size, self.tile_size))
            photo = ImageTk.PhotoImage(resized_tile, master=self.root)
            self.photo_image_refs.append(photo)
            tile_label.config(image=photo, highlightbackground=self.unselected_color)

        self.prompt_label.config(text=self.challenge.prompt)
        self.result_label.config(text=f"Score: {self.score_correct}/{self.score_attempts}")


    def submit(self):
        """
        FUNCTION NAME: submit

        DESCRIPTION: This function submits the selected tiles for grading and updates the score.

        INPUTS: None

        OUTPUTS: GradeResult, or None if there is nothing to grade
        """
        if self.challenge is None or self.submitted:
            return None

        selected_indices = sorted(self.selected_tiles)
        grade_result = self.env.grade(self.challenge, selected_indices)
        self.submitted = True
        self.score_attempts += 1
        if grade_result.passed:
            self.score_correct += 1

        status = "Passed!" if grade_result.passed else "Failed."
        self.result_label.config(
            text=f"{status} {grade_result.tiles_correct}/{len(self.tile_labels)} tiles correct. "
                 f"Score: {self.score_correct}/{self.score_attempts}"
        )
        return grade_result


    def toggle_tile_selection(self, tile_index):
        """
        FUNCTION NAME: toggle_tile_selection

        DESCRIPTION: This function toggles the selection state of a tile when it is clicked.

        INPUTS: tile_index (int)

        OUTPUTS: None
        """
        # Tiles are locked once the challenge has been submitted
        if self.submitted:
            return

        if tile_index in self.selected_tiles:
            self.selected_tiles.remove(tile_index)
            color = self.unselected_color
        else:
            self.selected_tiles.add(tile_index)
            color = SELECTED_COLOR
        self.tile_labels[tile_index].config(highlightbackground=color)


    def run(self):
        """
        FUNCTION NAME: run

        DESCRIPTION: This function starts the main loop of the GUI application.

        INPUTS: None

        OUTPUTS: None
        """
        self.root.mainloop()


#############################################################
#   FUNCTIONS
#############################################################
