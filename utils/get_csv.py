import os
import glob

import camelot
from PyPDF2 import PdfReader

def get_csv_tables(labels_path='./runs/detect/predict/labels', pdf_paths ="./datasets/pdfs/", tables_path="./datasets/tables"):
    'Function for extraction of tables in TXT format and excel file'
    pdf_paths = pdf_paths
    labels_path = labels_path
    tables_path=  tables_path
    #tables_path= "../llama-v8/datasets/truth_tables"
    csv_path=f"{tables_path}/text"
    if not os.path.exists(tables_path):
        os.makedirs(tables_path)
    if not os.path.exists(csv_path):
        os.makedirs(csv_path)

    labels= sorted(os.listdir(labels_path))
    for label in labels:
        print(label)
        pdf_name, page = os.path.splitext(label)[0].split('-')
        reader = PdfReader(f"{pdf_paths}/{pdf_name}.pdf")
        box = reader.pages[int(page)-1].mediabox

        h= box.height
        w= box.width  

        with open(f"{labels_path}/{label}",'r') as lf:
            lines = lf.readlines()
        for i,l in enumerate(lines):    
            if l[0] == '0' or l[0] == 0:
                x1n, y1n, x2n, y2n = list(l[2:].split())

                x1= int((float(x1n)- 0.5 * float(x2n)) * int(w))
                y1= int((float(y1n) - 0.5 * float(y2n)) * int(h))

                x2= int((x1 + float(x2n) * int(w)))
                y2= int((y1 + float(y2n) * int(h)))

                try: 
                    tables = camelot.read_pdf(f"{pdf_paths}/{pdf_name}.pdf", flavor='stream', table_areas=[f"{x1}, {h-y1}, {x2}, {h-y2}"], pages= str(int(page)),flag_size=True)
                    tables[0].to_csv(f"{csv_path}/{pdf_name}-{page}-{i}.txt")
                    tables[0].df.to_excel(f"{tables_path}/{pdf_name}-{page}-{i}.xlsx",index=False,header=False)
                except:
                    print(f"Table {pdf_name}-{page}-{i} couldnt be saved in excel")
                    continue
