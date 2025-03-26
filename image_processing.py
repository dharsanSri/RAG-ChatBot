from PIL import Image
import pytesseract
import io

def extract_text_from_image(uploaded_file):
    image = Image.open(io.BytesIO(uploaded_file.getvalue()))
    text = pytesseract.image_to_string(image)
    return text.strip()
