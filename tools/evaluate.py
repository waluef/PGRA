"""Aggregate results."""

import argparse
import os
import statistics


def read_data(filename):
    parsed = {}
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("stop_epoch"):
                continue
            key, values_str = line.split(":", 1)
            parsed[key] = [float(x) for x in values_str.strip().split("\t")]
    return parsed


def calculate_mean_std(values):
    mean = round(statistics.mean(values) * 100, 2)
    std = round(statistics.pstdev(values) * 100, 2)
    return mean, std


def append_summary(filename, results):
    with open(filename, "a", encoding="utf-8") as f:
        f.write("\n\nSummary:\n")
        for key, (mean, std) in results.items():
            std_str = "{:.3g}".format(std).ljust(4, "0")
            f.write("- **{}**: {:.2f} ± {}\n".format(key, mean, std_str))


def main():
    parser = argparse.ArgumentParser(
        description="Compute mean +/- std accuracy across runs")
    parser.add_argument("--path", required=True,
                        help="Directory containing result.txt")
    args = parser.parse_args()

    source = os.path.join(args.path, "result.txt")
    if not os.path.isfile(source):
        raise FileNotFoundError("result.txt not found in {}".format(args.path))

    data = read_data(source)
    print("Path:", source)
    print("Summary:")
    results = {}
    for key, values in data.items():
        mean, std = calculate_mean_std(values)
        results[key] = (mean, std)
        std_str = "{:.3g}".format(std).ljust(4, "0")
        print("- **{}**: {:.2f} ± {}".format(key, mean, std_str))
    append_summary(source, results)
    print("Summary appended to:", source)


if __name__ == "__main__":
    main()
