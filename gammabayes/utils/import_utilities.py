import importlib

# Function to dynamically import the desired function
def dynamic_import(module_path: str, object_name: str):
    """
    Dynamically imports a specified object (e.g., function, class) from a given module.

    Args:
        module_path (str): The path to the module from which to import the object.
        object_name (str): The name of the object to import from the module.

    Returns:
        object: The imported object if successful, or None if an error occurred.

    Raises:
        ImportError: If the module cannot be imported.
        AttributeError: If the object cannot be found in the module.
    """

    try:
        # Import the module
        module = importlib.import_module(module_path)
        
        # Fetch the function from the module
        func = getattr(module, object_name)
        
        return func
    except (ImportError, AttributeError) as e:
        # Handle the error (module or function not found)
        print(f"Error importing {object_name}: {e}")
        return None