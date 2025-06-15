from nbclient import NotebookClient
from nbformat import read
import time
from datetime import datetime

NOTEBOOK_PATH = "/Users/natwat/Desktop/CPSC_Projects/Trader/scripts/Data_manager.ipynb"

def run_notebook(path):
    with open(path) as f:
        nb = read(f, as_version=4)
        client = NotebookClient(nb, timeout=600, kernel_name='python3')
        print(f"[{datetime.now()}] Running notebook: {path}")
        client.execute()
        print(f"[{datetime.now()}] Notebook finished.")

if __name__ == "__main__":

        run_notebook(NOTEBOOK_PATH)