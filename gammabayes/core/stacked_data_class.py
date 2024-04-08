from gammabayes import EventData


class StackedEventData(object):
    """A class to contain multiple observation datasets for combined analysis."""

    def __init__(self, collection_metadata: dict | str = None, *args : EventData):
        self.collection_metadata = collection_metadata
        self.other_classes = []

        for arg in args:
            if isinstance(arg, EventData):
                self.other_classes.append(arg)
            else:
                raise TypeError("All arguments besides 'general_info' must be instances of EventData")
