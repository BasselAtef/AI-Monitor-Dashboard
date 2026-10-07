# Threads handle concurrency; one worker process keeps the connection pool
# simple and avoids cross-process state. The pool in db.py bounds database
# connections, so raise --workers only alongside a matching pool max_size.
web: gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --threads 8 --timeout 60
