def configure():
    import os
    import importlib.resources

    filename = './config.ini'

    if os.path.isfile(filename):
        print("File config.ini already exists")
    else:
        # Access the file as a resource
        with importlib.resources.open_text("asap.resources", "config.ini") as file:
            g = file.read()
            
            f = open(filename, 'w')
            f.write(g)
            f.close()
            print("File config.ini created")
            file.close()
