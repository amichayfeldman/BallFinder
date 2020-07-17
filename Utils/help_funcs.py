import csv


def write_to_csv(csv_path, list_to_write):
    with open(csv_path, 'w') as file:
        wr = csv.writer(file, quoting=csv.QUOTE_ALL)
        wr.writerow(list_to_write)