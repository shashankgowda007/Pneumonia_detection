import argparse
import subprocess
import sys
from multiprocessing import Process

def run_web_interface():
    subprocess.run(["streamlit", "run", "src/web_interface.py"])

def run_api_server():
    subprocess.run(["python", "src/api_server.py"])

def run_training():
    subprocess.run(["python", "src/train.py"])

def main():
    parser = argparse.ArgumentParser(description="DocAssist QA System")
    parser.add_argument("--web", action="store_true", help="Run web interface")
    parser.add_argument("--api", action="store_true", help="Run API server")
    parser.add_argument("--train", action="store_true", help="Run training")
    parser.add_argument("--all", action="store_true", help="Run all services")
    
    args = parser.parse_args()
    
    processes = []
    
    if args.web or args.all:
        p = Process(target=run_web_interface)
        p.start()
        processes.append(p)
    
    if args.api or args.all:
        p = Process(target=run_api_server)
        p.start()
        processes.append(p)
    
    if args.train or args.all:
        p = Process(target=run_training)
        p.start()
        processes.append(p)
    
    for p in processes:
        p.join()

if __name__ == "__main__":
    main()
