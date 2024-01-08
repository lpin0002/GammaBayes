import importlib

# Function to dynamically import the desired function
def dynamic_import(module_path: str, object_name: str):
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