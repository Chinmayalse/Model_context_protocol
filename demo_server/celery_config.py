"""
Celery configuration for the Medical Report Processing application.
"""

import os
from celery import Celery

# Configure Redis connection
REDIS_HOST = os.environ.get('REDIS_HOST', 'localhost')
REDIS_PORT = os.environ.get('REDIS_PORT', '6379')
REDIS_DB = os.environ.get('REDIS_DB', '0')
REDIS_URL = f"redis://{REDIS_HOST}:{REDIS_PORT}/{REDIS_DB}"

# Configure Celery
celery_app = Celery(
    'mcp_tasks',
    broker=REDIS_URL,
    backend=REDIS_URL,
    include=['tasks']
)

# Configure Celery settings
celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    task_track_started=True,
    task_time_limit=3600,  # 1 hour time limit per task
    worker_max_tasks_per_child=100,  # Restart worker after 100 tasks
    worker_prefetch_multiplier=1,  # Fetch one task at a time
    task_acks_late=True,  # Acknowledge tasks after execution
    task_reject_on_worker_lost=True,  # Reject tasks if worker dies
    broker_connection_retry=True,
    broker_connection_retry_on_startup=True,
    broker_connection_max_retries=10,
    result_expires=86400,  # Results expire after 1 day
)

# Define task routes
celery_app.conf.task_routes = {
    'tasks.process_file_task': {'queue': 'file_processing'},
    'tasks.process_batch_task': {'queue': 'batch_processing'},
    'tasks.update_batch_after_process': {'queue': 'batch_updates'},
    'tasks.check_batch_status': {'queue': 'status_checks'},
    'tasks.ping': {'queue': 'default'},
}

# Define periodic tasks
celery_app.conf.beat_schedule = {
    'cleanup-expired-results': {
        'task': 'tasks.cleanup_expired_results',
        'schedule': 86400.0,  # Run once a day
    },
}

# For Windows compatibility
celery_app.conf.worker_pool = 'solo' if os.name == 'nt' else 'prefork'
