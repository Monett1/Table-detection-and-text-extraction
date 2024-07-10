import time

from utils.utils import convert_pdf, predict
from utils.get_csv import get_csv_tables


convert_pdf(pdf_path='./datasets/pdfs')

predict(test_images_path= 'datasets/images',weights='datasets/weights/best.pt')   

get_csv_tables(labels_path='./runs/detect/predict/labels', pdf_paths ="./datasets/pdfs", tables_path="./datasets/tables")




