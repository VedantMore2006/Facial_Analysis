# backend/test_worker.py
import time
import camera_worker

camera_worker.start_pipeline()
time.sleep(10)
camera_worker.stop_pipeline()