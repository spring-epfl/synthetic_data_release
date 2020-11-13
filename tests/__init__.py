import sys, os
path = os.path.dirname(__file__)
print(path)
if path not in sys.path:
    sys.path.append(path)