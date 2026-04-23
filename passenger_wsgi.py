"""
WSGI entry point for cPanel Python hosting (LiteSpeed/Passenger).
cPanel activates the correct virtualenv automatically before running this file.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import app as application
