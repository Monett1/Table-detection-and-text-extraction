import os
import fitz
import json
from pathlib import Path
import base64
import requests


from ultralytics import YOLO

from openai import OpenAI
from PyPDF2 import PdfReader
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from transformers import pipeline, AutoTokenizer

## Convert each of the pdf page to images for indexing

def convert_pdf(pdf_path='./datasets/pdfs'):
    '''
    Converts pdf pages into images

    pdf_path: The path of the pdfs whose pages are to be transformed into images
    '''
    pdf_file = sorted(os.listdir(pdf_path))
    
    if not os.path.exists("datasets/images"):
            os.makedirs("datasets/images")

    for pdf in pdf_file:
        _,extension= os.path.splitext(pdf)
        if extension != '.pdf':
            continue
        # Split the base name and extension
        output_directory_path, _ = os.path.splitext(pdf)

        
        # Open the PDF file
        pdf_document = fitz.open(f"{pdf_path}/{pdf}")

        position= output_directory_path.rfind('/') 
        if position != -1:

            # Iterate through each page and convert to an image
            for page_number in range(pdf_document.page_count):
                # Get the page
                page = pdf_document[page_number]

                # Convert the page to an image
                pix = page.get_pixmap()

                # Create a Pillow Image object from the pixmap
                image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

                # Save the image
                image.save(f"datasets/images/{output_directory_path[position+1:]}-{page_number + 1}.jpg") ##page numbre +1 ??
        else :
            # Iterate through each page and convert to an image
            for page_number in range(pdf_document.page_count):
                # Get the page
                page = pdf_document[page_number]

                # Convert the page to an image
                pix = page.get_pixmap()

                # Create a Pillow Image object from the pixmap
                image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

                # Save the image
                image.save(f"datasets/images/{output_directory_path}-{page_number + 1}.jpg")  ##page numbre +1 ??
        # Close the PDF file
        pdf_document.close()

    
def plot_images(image_paths):
    'Plot images extracted from pdf'

    images_shown = 0
    plt.figure(figsize=(16, 9))
    for img_path in image_paths:
        if os.path.isfile(img_path):
            image = Image.open(img_path)

            plt.subplot(3, 3, images_shown + 1)
            plt.imshow(image)
            plt.xticks([])
            plt.yticks([])

            images_shown += 1
            if images_shown >= 9:
                break


def make_folders(path="output"):
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def convert_bbox_coco2yolo(img_width, img_height, bbox):
    """
    Convert bounding box from COCO  format to YOLO format

    Parameters
    ----------
    img_width : int
        width of image
    img_height : int
        height of image
    bbox : list[int]
        bounding box annotation in COCO format: 
        [top left x position, top left y position, width, height]

    Returns
    -------
    list[float]
        bounding box annotation in YOLO format: 
        [x_center_rel, y_center_rel, width_rel, height_rel]
    """
    
    # YOLO bounding box format: [x_center, y_center, width, height]
    # (float values relative to width and height of image)
    x_tl, y_tl, w, h = bbox

    dw = 1.0 / img_width
    dh = 1.0 / img_height

    x_center = x_tl + w / 2.0
    y_center = y_tl + h / 2.0

    x = x_center * dw
    y = y_center * dh
    w = w * dw
    h = h * dh

    return [x, y, w, h]

def convert_coco_json_to_yolo_txt(output_path, json_file):
    '''
    Converts a json file whose annotations are in COCO format 
    to a number of text files in YOLO format equal to the number 
    of labeled images  
    
    '''

    path = make_folders(output_path)

    with open(json_file, 'r+') as f:
        json_data = json.load(f)


    for image in tqdm(json_data["images"], desc="Annotation txt for each iamge"):
        img_id = image["id"]
        img_name_extension = image["file_name"]
        img_width = image["width"]
        img_height = image["height"]

        img_name,_= os.path.splitext(img_name_extension)
        img_name=img_name.split('-')[0]+'-'+ str(int(img_name.split('-')[1])-1)


        anno_in_image = [anno for anno in json_data["annotations"] if anno["image_id"] == img_id]
        anno_txt = os.path.join(output_path, img_name + ".txt")
        with open(anno_txt, "w") as f:
            for anno in anno_in_image:
                category = anno["category_id"]-1
                bbox_COCO = anno["bbox"]
                x, y, w, h = convert_bbox_coco2yolo(img_width, img_height, bbox_COCO)
                if anno == anno_in_image[-1]:
                    f.write(f"{category} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")
                else:
                    f.write(f"{category} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")

def predict(test_images_path= 'datasets/images',weights= 'datasets/weights/best.pt'):
    test_images_path= test_images_path
    images_names= sorted(os.listdir(test_images_path))
    weights = weights
    model= YOLO(weights)

    for image in images_names:
        model.predict(source=f"{test_images_path}/{image}", conf=0.7, iou=0.8, save_txt= True)

def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')
