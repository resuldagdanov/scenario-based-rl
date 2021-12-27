import os
import sys

# to add the parent "agents" folder to sys path and import models
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from utils.db import DB
db = DB()
db.initialize_tables()
db.close()