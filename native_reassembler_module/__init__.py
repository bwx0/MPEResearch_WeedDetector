import os

if os.name == "nt":  # Windows
    for directory in os.environ.get("PATH", "").split(os.pathsep):
        try:
            os.add_dll_directory(directory)
        except:
            pass
