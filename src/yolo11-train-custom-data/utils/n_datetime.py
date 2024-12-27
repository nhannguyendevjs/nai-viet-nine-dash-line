from datetime import datetime


class NDateTime:
    """DateTime utility"""

    def __init__(self):
        self.timezone = None

    def get_current_timestamp(self):
        """Get current timestamp in milliseconds."""
        return int(datetime.now().timestamp() * 1000)
