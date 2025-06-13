import pytesseract
from PIL import Image
import sys
import json 
import os.path

def splitLines(text):
    lines = text.splitlines()
    result = []
    current_segment = []
    
    for line in lines:
        if line.strip() == '':
            if current_segment:
                result.append('\n'.join(current_segment))
                current_segment = []
        else:
            current_segment.append(line)
    
    if current_segment:
        result.append('\n'.join(current_segment))

    return result

def cleanResult(text):
    text = splitLines(text)
    for i in range(len(text)):
        text[i] = text[i].replace('\n', ' ')
    
    return text

def output(result):
    for line in result:
        print(line)

def getImagePath():
        try:
            # Checks if the address is provided or not
            if len(sys.argv) < 2:
                print(json.dumps({'error':"No address provided"}))
                sys.exit(1)

            image_path = sys.argv[1]
            # Checks if the file exists or not
            if not os.path.isfile(image_path):
                print(json.dumps({'error':f"File not found at {image_path}"}))
                sys.exit(1)
            
            # Checks if the input is a valid image or not
            extensions = ['.png', '.jpg', '.jpeg']
            if not image_path.lower().endswith(extensions):
                print(json.dumps({'error':"File format is not supported"}))
                sys.exit(1)

            return image_path

        except ValueError:
            print(json.dumps({'error':'Invalid input'}))
            sys.exit(1)

if __name__ == '__main__':
    image_path = getImagePath()
    image = Image.open(image_path)

    # Performing OCR
    text = pytesseract.image_to_string(image)
    result = cleanResult(text)
    
    # Output
    output(result)
    sys.stdout.flush()