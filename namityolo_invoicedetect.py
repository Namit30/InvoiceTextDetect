# -*- coding: utf-8 -*-
"""Untitled0.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1Cnw8r5y-mFpQqy1Ph4BQPYffjP2me0g-
"""

!nvidia-smi

!pip install ultralytics

from ultralytics import YOLO
import os
from IPython.display import display, Image
from IPython import display
display.clear_output()
!yolo checks
# model = YOLO("yolov8s.pt") # load the model
# results = model.train(data="coco128.yaml", epochs=100)
# results = model("./image.png")

!mkdir pdfs

!sudo apt-get install poppler-utils

!pip install -qqq pdf2image==1.16.3 --progress-bar off
!pip install -qqq pypdf==3.15.0 --progress-bar off
!pip install pdfplumber

from pdf2image import convert_from_path
invoice = convert_from_path("/content/pdfs/invoice2.pdf", dpi=300)
invoice[0]

!pip install tesseract --progress-bar off
! pip install Pillow --progress-bar off
! pip install pytesseract --progress-bar off
!pip install ocrmypdf --progress-bar off

import os
import re
import pdfplumber
import ocrmypdf


input_path = r'/content/pdfs/invoice2.pdf'
output_path = r'/content/pdfs/outputinvoice2.pdf'

def extract_invoice_details(text):
    # Extract the invoice number
    invoice_number_match = re.search(r'Invoice Number\s*:\s*(\S+)', text)
    if not invoice_number_match:
        invoice_number_match = re.search(r'Invoice Number\s*(\S+)', text)
    invoice_number = invoice_number_match.group(1).strip() if invoice_number_match else None

    # Extract the total amount due
    total_amount_due_match = re.search(r'Total Due\s*:\s*(\S+)', text)
    if not total_amount_due_match:
        total_amount_due_match = re.search(r'Total Due\s*(\S+)', text)
    total_amount_due = total_amount_due_match.group(1).strip() if total_amount_due_match else None

    # Extract the invoice date
    invoice_date_match = re.search(r'Invoice Date\s*:\s*([A-Za-z]+\s+\d{1,2},\s+\d{4})', text)
    if not invoice_date_match:
        invoice_date_match = re.search(r'Invoice Date\s*([A-Za-z]+\s+\d{1,2},\s+\d{4})', text)
    invoice_date = invoice_date_match.group(1).strip() if invoice_date_match else None

    # Extract the due date
    due_date_match = re.search(r'Due Date\s*:\s*([A-Za-z]+\s+\d{1,2},\s+\d{4})', text)
    if not due_date_match:
        due_date_match = re.search(r'Due Date\s*([A-Za-z]+\s+\d{1,2},\s+\d{4})', text)
    due_date = due_date_match.group(1).strip() if due_date_match else None

    return invoice_number, total_amount_due, invoice_date, due_date

def process_pdf(input_path, use_ocr=False):
    try:
        if use_ocr:
            ocrmypdf.ocr(input_path, output_path, force_ocr=True)
            path = output_path
        else:
            path = input_path

        with pdfplumber.open(path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                text = page.extract_text()
                if text:
                    print(f"Text from Page {page_num + 1}:")
                    print(text)

                    # Extract invoice details
                    invoice_number, total_amount_due, invoice_date, due_date = extract_invoice_details(text)

                    # Print the extracted information
                    print('Invoice Number:', invoice_number)
                    print('Total Amount Due:', total_amount_due)
                    print('Invoice Date:', invoice_date)
                    print('Due Date:', due_date)

                    # If all details are found, no need to process further
                    if all([invoice_number, total_amount_due, invoice_date, due_date]):
                        return

                    # Visual debugging
                    img = page.to_image()
                    img.draw_rects(page.extract_words())
                    img_path = f'page_{page_num + 1}_{"ocr" if use_ocr else "original"}.png'
                    img.save(img_path)
                    print(f"Saved visual debug image for page {page_num + 1} at {img_path}")

    except FileNotFoundError:
        print(f"The file {input_path} was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

# Process the PDF without OCR first
process_pdf(input_path)

# If needed, process the PDF with OCR
process_pdf(input_path, use_ocr=True)

# !yolo predict model=yolov8n.pt source="https://ultralytics.com/images/bus.jpg"

# !pip install roboflow

# from roboflow import Roboflow
# rf = Roboflow(api_key="nnI1WMv1fj7eWnJGp35H")
# project = rf.workspace("namit-nbvnh").project("namit_ocr")
# version = project.version(1)
# dataset = version.download("yolov8")
# !pip install roboflow

# from roboflow import Roboflow
# rf = Roboflow(api_key="nnI1WMv1fj7eWnJGp35H")
# project = rf.workspace("invoicemodelisation").project("invoiceproject")
# version = project.version(5)
# dataset = version.download("yolov8")

# !pip install roboflow

# from roboflow import Roboflow
# rf = Roboflow(api_key="nnI1WMv1fj7eWnJGp35H")
# project = rf.workspace("namit-nbvnh").project("namit_ocr")
# version = project.version(2)
# dataset = version.download("yolov8")

# !pip install roboflow

# from roboflow import Roboflow
# rf = Roboflow(api_key="nnI1WMv1fj7eWnJGp35H")
# project = rf.workspace("namit-nbvnh").project("namit_ocr")
# version = project.version(4)
# dataset = version.download("yolov8")


# !pip install roboflow

# from roboflow import Roboflow
# rf = Roboflow(api_key="nnI1WMv1fj7eWnJGp35H")
# project = rf.workspace("namit-nbvnh").project("namit_ocr")
# version = project.version(3)
# dataset = version.download("yolov8")

!pip install roboflow

from roboflow import Roboflow
rf = Roboflow(api_key="nnI1WMv1fj7eWnJGp35H")
project = rf.workspace("namit-nbvnh").project("namit_ocr")
version = project.version(5)
dataset = version.download("yolov8")

from ultralytics import YOLO

# Load a model
model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)

# Train the model with 2 GPUs
results = model.train(data="/content/Namit_ocr-5/data.yaml", epochs=40, imgsz=640,batch = 8)

import glob
from IPython.display import Image, display

for image_path in glob.glob(f'/content/runs/detect/train2/*.jpg'):
  display(Image(filename=image_path, height=600))
  print("\n")

# Import necessary libraries
from pdf2image import convert_from_path
import os

# Define paths
input_path = r'/content/pdfs/invoice2.pdf'
output_folder = r'/content/pdfs/'

# Ensure the output folder exists
os.makedirs(output_folder, exist_ok=True)

# Function to convert PDF to images and save them in the specified folder
def convert_pdf_to_images(pdf_path, output_folder):
    images = convert_from_path(pdf_path, dpi=300)
    image_paths = []
    for i, image in enumerate(images):
        image_path = os.path.join(output_folder, f'invoice_page_{i + 1}.jpg')
        image.save(image_path, 'PNG')
        image_paths.append(image_path)
    return image_paths

# Convert the PDF to images and save them
image_paths = convert_pdf_to_images(input_path, output_folder)
print("Converted PDF pages to images:", image_paths)

# Display the first image
from IPython.display import display, Image
display(Image(filename=image_paths[0]))

# Process the PDF without OCR first (Optional, you can skip this if not needed)
# process_pdf(input_path)

# If needed, process the PDF with OCR (Optional, you can skip this if not needed)
# process_pdf(input_path, use_ocr=True)

# Continue with the rest of your code...

import locale
def getpreferredencoding(do_setlocale = True):
    return "UTF-8"
locale.getpreferredencoding = getpreferredencoding
!yolo task=detect mode=val model=/content/runs/detect/train2/weights/best.pt data={dataset.location}/data.yaml

!yolo task=detect mode=predict model=/content/runs/detect/train2/weights/best.pt source="https://i.postimg.cc/RZPZWbLy/Quality-Hosting-page-0001.jpg"

!yolo task=detect mode=val model=/content/runs/detect/train/weights/best.pt data={dataset.location}/data.yaml

!yolo task=detect mode=predict model=/content/runs/detect/train4/weights/last.pt source="https://i.postimg.cc/9MWDFZz2/Screenshot-2024-07-23-155059.jpg"