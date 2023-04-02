from mlflow import log_metric
from random import random, randint

if __name__ == "__main__":
    for i in range(100):
        log_metric("foo", random())
        log_metric("foo", randint(0, 100))