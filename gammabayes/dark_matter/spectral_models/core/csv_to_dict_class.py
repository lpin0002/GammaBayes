import pandas as pd
import numpy as np

class CSVDictionary:
    def __init__(self, filepath, delimiter=' ', numeric_type=np.float128):
        self.df = pd.read_csv(filepath, delimiter=delimiter)
        self._convert_numeric_columns(numeric_type)

        self._create_dict()



    def _convert_numeric_columns(self, numeric_type):
        """Converts all columns that can be converted to the specified numeric type."""
        for column in self.df.columns:
            # Attempt to convert each column to the specified numeric type.
            # If a column contains non-numeric data, it will remain unchanged.
            self.df[column] = pd.to_numeric(self.df[column], errors='ignore', downcast='float')
            if self.df[column].dtype == np.number or self.df[column].dtype == np.float_:
                self.df[column] = self.df[column].astype(numeric_type)
        

    def _create_dict(self):
        self.dict_object = self.df.to_dict('list')

        try:
            del self.dict_object['Unnamed: 0']
        except:
            pass

        for key in self.dict_object:
            self.dict_object[key] = np.asarray(self.dict_object[key])

    def __delitem__(self, key):
        """Allows deletion of items using dictionary-like syntax."""
        if key in self.dict_object:
            del self.dict_object[key]
        else:
            raise KeyError(f"Key '{key}' not found in the dictionary.")


    def __getitem__(self, key):
        """Allows column-based access like obj['column_name'] to get all values under that column."""
        return self.dict_object[key]
        
    def __setitem__(self, key, values):
        """Allows setting or updating column values."""
        # This simplistic implementation assumes 'values' is a complete list of column values.
        # You might need more complex handling depending on your exact requirements.
        self.dict_object[key] = values

        
    def keys(self):
        """Returns a list of column headers."""
        return self.dict_object.keys()
    
    def values(self):
        """Returns a list of NumPy arrays with the values for each column."""
        return self.dict_object.values()
    
    def items(self):
        """Returns a list of tuples, each tuple being (column_header, column_values_as_numpy_array)."""
        return self.dict_object.items()
    

    def pop(self, key, default=None):
        """Pops a column from the DataFrame, returning the column values.
        
        If the column does not exist and a default value is specified, returns the default value.
        Otherwise, raises a KeyError if the column does not exist and no default is provided.
        """
        if key in self.dict_object:
            return self.dict_object[key]
        else:
            if default is not None:
                return default
            else:
                raise KeyError(f"Column '{key}' does not exist.")


