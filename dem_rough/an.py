import csv

def check_csv_file(file_name):
    with open(file_name, 'r') as file:
        csv_reader = csv.reader(file)
        row_count = sum(1 for row in csv_reader)
        if row_count > 0:
            print("CSVファイルにはデータが書き込まれています。")
        else:
            print("CSVファイルは空です。")

# ファイル名を指定してCSVファイルをチェック
check_csv_file('aaa.csv')
