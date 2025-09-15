import pandas as pd

class PerformanceMonitor:
    """
    Collects and stores simulation data at each step for later analysis.
    """

    def __init__(self):
        """
        Initializes the monitor with a list to store records.
        """
        self.records = []

    def record(self, **kwargs):
        """
        Records a single timestep of data.

        :param kwargs: A dictionary of data points for the current step
                       (e.g., episode, step, reward, action, etc.).
        """
        self.records.append(kwargs)

    def get_dataframe(self):
        """
        Converts the collected records into a pandas DataFrame.

        :return: A DataFrame containing all simulation data, or an empty
                 DataFrame if no records were collected.
        """
        if not self.records:
            return pd.DataFrame()
        return pd.DataFrame(self.records)

