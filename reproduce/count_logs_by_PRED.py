import re
from collections import defaultdict

import fire


def main(name, round_num=1, cutoff=50, show_num=50):
    # Define input log file
    threshold = None
    round_num = int(round_num)
    cutoff = int(cutoff)
    show_num = int(show_num)
    log_file = name
    if round_num == 1:
        threshold = 180
    elif round_num == 2:
        threshold = 289
    elif round_num == 3:
        threshold = 316
    else:
        print("Input the correct round value.")

    # Dictionary to store counts of identical fractions + ML predicted value
    fraction_counts = defaultdict(int)
    fraction_data = []

    pattern = re.compile(r"^(\d+)\s+(\[.*?\])\s+ML predicted:\s+([\d\.]+)")

    # Read and process the log file
    with open(log_file, "r") as file:
        for line in file:
            # Extract first number (ID), fractions, and ML predicted value
            # match = re.match(r"(\d+) (\[.*?\]) .* ML predicted: (.+)", line)
            match = pattern.match(line)
            if match:
                first_number = int(match.group(1))
                fractions = match.group(2)
                ml_predicted = float(match.group(3))
                ml_predicted = round(ml_predicted, 6)

                # Filter out rows where the first number < 180
                if first_number >= threshold:
                    # print(first_number)
                    key = (fractions, ml_predicted)
                    fraction_counts[key] += 1  # Count occurrences

    # Convert dictionary to a sortable list
    for (fractions, ml_predicted), count in fraction_counts.items():
        fraction_data.append((fractions, ml_predicted, count))

    # Sort the results by ML predicted value (descending order)
    fraction_data.sort(key=lambda x: x[1], reverse=True)

    count_stop = 0
    print("Identical Fractions & ML Predictions Count (Sorted by ML predicted value):")
    for fractions, ml_predicted, count in fraction_data:
        if count_stop > show_num:
            break
        if count > cutoff:
            print(f"{fractions} ML predicted: {ml_predicted:.6f} → Count: {count}")
            if count > 100:
                count_stop += 1
                print("↑↑↑↑↑↑↑↑This fractions is chosen for experiments↑↑↑↑↑↑↑↑")


if __name__ == "__main__":
    fire.Fire(main)
