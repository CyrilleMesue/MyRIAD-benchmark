# Imports
from modules.utils import save_object, save_json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sys

from modules.exception import CustomException
from modules.logger import logging


class BaseConfigurable:
    """
    Base class providing common configuration options and utility methods
    for controlling plot display, result saving, and logging output.

    Attributes:
        show_plots (bool): Whether to display plots.
        save_results (bool): Whether to save output data.
        verbose (bool): Whether to print/log output messages.
    """

    def __init__(self, show_plots=True, save_results=False, verbose=True):
        """
        Initialize the configuration flags.

        Args:
            show_plots (bool): Flag to control display of plots. Default is True.
            save_results (bool): Flag to control saving of results. Default is False.
            verbose (bool): Flag to control printing/logging. Default is True.
        """
        self.show_plots = show_plots
        self.save_results = save_results
        self.verbose = verbose

    def print(self, message):
        """
        Print a message to the console if verbose mode is enabled.

        Args:
            message (str): The message to print.

        Raises:
            CustomException: If printing fails.
        """
        try:
            if self.verbose:
                print(message)
        except Exception as e:
            logging.info(CustomException(e, sys))
            raise CustomException(e, sys)

    def maybe_show(self, fig=None):
        """
        Show a matplotlib figure if show_plots is enabled.

        Args:
            fig (matplotlib.figure.Figure): The figure to display.

        Raises:
            CustomException: If showing the figure fails.
        """
        try:
            if self.show_plots and fig is not None:
                fig.show()
            else:
                fig.close()
        except Exception as e:
            logging.info(CustomException(e, sys))
            raise CustomException(e, sys)

    def maybe_save(self, data, filename, object_type: str = "object", data_name: str = None):
        """
        Save data to file if save_results is enabled.

        Args:
            data: The data to save (could be object, JSON, DataFrame, or Figure).
            filename (str): Destination file path.
            object_type (str): Type of data - "object", "json", "table", or "figure".
            data_name (str): Optional name of the data (for logging purposes).

        Raises:
            CustomException: If saving fails.
        """
        try:
            if self.save_results:
                # Save based on type
                if object_type == "object":
                    save_object(filename, data)
                elif object_type == "json":
                    save_json(filename, data)
                elif object_type == "table":
                    data.to_csv(filename, index=False)
                elif object_type == "figure":
                    data.savefig(filename, bbox_inches='tight')
                else:
                    raise ValueError(f"Unsupported object_type: {object_type}")

                logging.info(f"Saved {data_name or 'data'} to {filename}")
        except Exception as e:
            logging.info(CustomException(e, sys))
            raise CustomException(e, sys)
