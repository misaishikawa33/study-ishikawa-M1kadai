import pandas as pd


class AicsExcelLoader:
    def __init__(self, file_path):
        self.sheet_names = pd.ExcelFile(file_path).sheet_names
        print(f"input sheets: {self.sheet_names}")

        if len(self.sheet_names) == 1:
            self.data = pd.read_excel(file_path)
        else:
            self.data = pd.read_excel(file_path, sheet_name=None)
