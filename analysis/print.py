#!/usr/bin/env python3
import pandas as pd
import sys

def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <input.csv>", file=sys.stderr)
        sys.exit(1)
    
    csv_file = sys.argv[1]
    
    # Read the CSV
    df = pd.read_csv(csv_file)
    
    # Group by model and calculate means for numeric columns
    grouped_means = df.groupby('model').mean(numeric_only=True)
    
    # Print the results
    print("Mean values grouped by model:")
    print("=" * 80)
    print(grouped_means.to_string())
    print()
    
    # Optionally, save to CSV
    # grouped_means.to_csv("model_means.csv")

if __name__ == "__main__":
    main()