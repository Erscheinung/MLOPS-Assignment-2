#!/bin/bash

# Install or update dependencies
pip install -r requirements.txt

# Run your Flask app
gunicorn --bind=0.0.0.0 --timeout 600 app:app