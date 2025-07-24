import os
from pathlib import Path
import pdfplumber


def extract_text_from_pdf(pdf_path, txt_path):
    with pdfplumber.open(pdf_path) as pdf:
        all_text = ""
        for page in pdf.pages:
            all_text += page.extract_text() or ""
            all_text += "\n"
    os.makedirs(os.path.dirname(txt_path), exist_ok=True)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(all_text)
    print(f"Saved text: {txt_path}")


def main():
    data_dir = Path("./data")
    for key in ["ai", "ai_product"]:
        pdf_path = data_dir / f"{key}_plan.pdf"
        txt_path = data_dir / f"{key}_plan.txt"
        if pdf_path.exists():
            extract_text_from_pdf(str(pdf_path), str(txt_path))
        else:
            print(f"PDF not found: {pdf_path}")


if __name__ == "__main__":
    main()
