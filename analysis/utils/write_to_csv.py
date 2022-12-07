import csv

'''
file_name 文件名称
headers 表头数组
'''


def write_csv_hearders(file_name, headers):
    with open(file_name, 'w', newline='', encoding="utf-8") as f:
        f_csv = csv.writer(f)
        f_csv.writerow(headers)


'''
file_name 文件名称
rows 数据([[],[]])
'''


def write_csv_rows(file_name, rows):
    with open(file_name, 'a+', newline='', encoding='utf-8') as f:
        f_csv = csv.writer(f)
        f_csv.writerows(rows)
    f.close()
