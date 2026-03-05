class BaseFeature:
    """
    Base class for all behavioral features.
    """

    def __init__(self, name, history_size=300):
        """
        Initialize feature.

        Parameters
        ----------
        name : str
            Feature name
        history_size : int
            Maximum stored history values
        """
        self.name = name
        self.history = []
        self.history_size = history_size

    def compute(self, landmarks, frame_buffer, timestamp):
        """
        Compute feature value.

        Must be implemented by child classes.
        """
        raise NotImplementedError

    def update_history(self, value):
        """
        Store feature value in history.
        """
        self.history.append(value)

        if len(self.history) > self.history_size:
            self.history.pop(0)

    def get_latest(self):
        """
        Get most recent value.
        """
        if not self.history:
            return None

        return self.history[-1]
