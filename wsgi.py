import sys
import os

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import your Streamlit app
from web_interface_v2 import app

# Create WSGI application
application = app.server 