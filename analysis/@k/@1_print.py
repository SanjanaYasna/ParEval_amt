#!/usr/bin/env python3
import csv
import sys

def main():
    filepath = sys.argv[1] if len(sys.argv) > 1 else "results.csv"
    
    with open(filepath, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    # Find all @1 columns
    at1_cols = [col for col in rows[0].keys() if col.endswith("@1")]
    
    # Print header
    print(f"{'model':<15}", end="")
    for col in at1_cols:
        print(f"{col:>20}", end="")
    print()
    print("-" * (15 + 20 * len(at1_cols)))
    
    # Print rows
    for row in rows:
        print(f"{row['model']:<15}", end="")
        for col in at1_cols:
            val = float(row[col])
            print(f"{val:>20.4f}", end="")
        print()

if __name__ == "__main__":
    main()