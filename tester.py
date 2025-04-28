#!/usr/bin/env python3
"""
test_infer.py

Simple interactive tester for the /infer endpoint.
"""

import requests
import json

API_URL = "http://127.0.0.1:8000/infer"

def test_known():
    tid = input("Enter a precomputed transcript ID (e.g. TX_0001): ").strip()
    payload = {"transcript_id": tid}
    print("\nRequest payload:")
    print(json.dumps(payload, indent=2))
    r = requests.post(API_URL, json=payload)
    print("\nResponse:")
    print(json.dumps(r.json(), indent=2))

def test_custom():
    seq = input("Paste your RNA sequence (A/C/G/U): ").strip().upper()
    chr_ = input("Chromosome (e.g. chr1): ").strip()
    start = int(input("Start position (e.g. 100000): ").strip())
    end   = int(input("End position   (e.g. 100050): ").strip())
    payload = {
        "sequence": seq,
        "chr": chr_,
        "start": start,
        "end": end
    }
    print("\nRequest payload:")
    print(json.dumps(payload, indent=2))
    r = requests.post(API_URL, json=payload)
    print("\nResponse:")
    print(json.dumps(r.json(), indent=2))

def main():
    print("=== lncWrapper /infer Tester ===")
    print("1) Test known transcript")
    print("2) Test custom sequence")
    choice = input("Select mode (1 or 2): ").strip()
    if choice == "1":
        test_known()
    elif choice == "2":
        test_custom()
    else:
        print("Invalid choice. Exiting.")

if __name__ == "__main__":
    main()
