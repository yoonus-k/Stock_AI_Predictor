import os

def add_init_files(directory):
    """Adds __init__.py files to all subdirectories within the given directory."""
    for root, dirs, files in os.walk(directory):
        for dir_name in dirs:
            init_file_path = os.path.join(root, dir_name, '__init__.py')
            if not os.path.exists(init_file_path):
                with open(init_file_path, 'w') as init_file:
                    pass  # Create an empty file

if __name__ == "__main__":
    # the main directory as the current working directory
    target_directory = os.getcwd()  # or specify a different directory
    
    add_init_files(target_directory)
    print(f"__init__.py files added to all subdirectories within '{target_directory}'.")
    
    # then install the package in the main directory
    os.system(f"pip install -e {target_directory}")
    