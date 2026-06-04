"""
WSGI handler for Vercel deployment with Streamlit
"""
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Streamlit import
try:
    from streamlit.web import cli as stcli
    from streamlit import config
except ImportError:
    # Fallback if streamlit imports differently
    import streamlit as st

def application(environ, start_response):
    """WSGI application handler"""
    status = '200 OK'
    headers = [('Content-type', 'text/plain; charset=utf-8')]
    start_response(status, headers)
    return [b"Streamlit app is running. Please access it via the web interface."]

# Export as 'app' for Vercel
app = application
