import pandas as pd
import numpy as np

class CSVDictionary:
    def __init__(self, filepath, delimiter=' ', numeric_type=float):
        self.df = pd.read_csv(filepath, delimiter=delimiter)
        self._convert_numeric_columns(numeric_type)

    def _convert_numeric_columns(self, numeric_type):
        """Converts all columns that can be converted to the specified numeric type."""
        for column in self.df.columns:
            # Attempt to convert each column to the specified numeric type.
            # If a column contains non-numeric data, it will remain unchanged.
            self.df[column] = pd.to_numeric(self.df[column], errors='ignore', downcast='float')
            if self.df[column].dtype == np.number or self.df[column].dtype == np.float_:
                self.df[column] = self.df[column].astype(numeric_type)

    
    def __getitem__(self, column_name):
        """Allows column-based access like obj['column_name'] to get all values under that column."""
        if column_name in self.df:
            return self.df[column_name].values
        else:
            raise KeyError(f"Column {column_name} does not exist.")
        
    def __setitem__(self, column_name, values):
        """Allows setting or updating column values."""
        # This simplistic implementation assumes 'values' is a complete list of column values.
        # You might need more complex handling depending on your exact requirements.
        self.df[column_name] = values

        
    def keys(self):
        """Returns a list of column headers."""
        return self.df.columns.tolist()
    
    def values(self):
        """Returns a list of NumPy arrays with the values for each column."""
        return [self.df[column].values for column in self.df.columns]
    
    def items(self):
        """Returns a list of tuples, each tuple being (column_header, column_values_as_numpy_array)."""
        return [(column, self.df[column].values) for column in self.df.columns]
    

    def pop(self, column_name, default=None):
        """Pops a column from the DataFrame, returning the column values.
        
        If the column does not exist and a default value is specified, returns the default value.
        Otherwise, raises a KeyError if the column does not exist and no default is provided.
        """
        if column_name in self.df:
            values = self.df[column_name].values
            self.df.drop(column_name, axis=1, inplace=True)
            return values
        else:
            if default is not None:
                return default
            else:
                raise KeyError(f"Column '{column_name}' does not exist.")


