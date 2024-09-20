#!/bin/bash
# Activate venv and send a notification to istream player about impending bandwidth change

source $(dirname $0)/env/bin/activate
python $(dirname $0)/send_notification.py "$@"