import time
import logging

def get_time(start, end):
    elapsed_time = end - start
    minutes = int(elapsed_time // 60)
    seconds = elapsed_time % 60
    logging.info(f"Execution time: {minutes} minutes {seconds:.2f} seconds")
    print(f"Execution time: {minutes} minutes {seconds:.2f} seconds")