# One worker, many threads. SQLite allows a single writer, so multiple worker
# processes would contend and raise "database is locked" under concurrent calls.
# Raise this only after moving DB_PATH to Postgres.
web: gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --threads 8 --timeout 60
