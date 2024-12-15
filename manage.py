#!/usr/bin/env python
"""Django's command-line utility for administrative tasks."""
import os
import sys


def main():
    """Run administrative tasks."""
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'sentiment_analysis_server.settings')
    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Couldn't import Django. Are you sure it's installed and "
            "available on your PYTHONPATH environment variable? Did you "
            "forget to activate a virtual environment?"
        ) from exc
    if len(sys.argv) == 1:
        sys.argv += ["runserver", "127.0.0.1:3110"]  # Lệnh runserver và địa chỉ + port
    elif sys.argv[1] == "runserver" and len(sys.argv) == 2:
        sys.argv += ["127.0.0.1:3110"]  # Chỉ thêm port nếu chưa có
    execute_from_command_line(sys.argv)


if __name__ == '__main__':
    main()
