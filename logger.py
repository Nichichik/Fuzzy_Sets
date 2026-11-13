import logging
from datetime import datetime

logging.basicConfig(
    filename='log.txt',
    filemode='a', 
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO,
    encoding='utf-8'
)

def log(message: str):
    logging.info(message)
    print(f"[LOG] {message}")
