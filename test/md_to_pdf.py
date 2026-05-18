import os
import sys
from fpdf import FPDF
from fpdf.enums import XPos, YPos

class PDF(FPDF):
    def header(self):
        pass

    def footer(self):
        self.set_y(-15)
        self.set_font('Malgun', '', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

def convert_md_to_pdf(md_path, pdf_path):
    if not os.path.exists(md_path):
        print(f"Error: {md_path} not found")
        return

    pdf = PDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_margins(left=15, top=15, right=15)
    
    font_path = "/mnt/c/Windows/Fonts/malgun.ttf"
    if os.path.exists(font_path):
        pdf.add_font('Malgun', '', font_path)
        pdf.set_font('Malgun', '', 11)
    else:
        print("Warning: Malgun Gothic font not found. Korean text may not render correctly.")
        pdf.set_font("helvetica", size=11)

    pdf.add_page()
    
    with open(md_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        if not line:
            pdf.ln(5)
            continue
            
        # Basic MD parsing
        if line.startswith('# '):
            pdf.set_font('Malgun', '', 18)
            pdf.cell(0, 12, line[2:], new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.set_font('Malgun', '', 11)
        elif line.startswith('## '):
            pdf.set_font('Malgun', '', 15)
            pdf.cell(0, 10, line[3:], new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.set_font('Malgun', '', 11)
        elif line.startswith('### '):
            pdf.set_font('Malgun', '', 13)
            pdf.cell(0, 8, line[4:], new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.set_font('Malgun', '', 11)
        elif line.startswith('![') and '](' in line:
            start = line.find('(') + 1
            end = line.find(')')
            img_path = line[start:end]
            full_img_path = os.path.join(os.path.dirname(md_path), img_path)
            if os.path.exists(full_img_path):
                # Scale image to fit width (180mm)
                pdf.image(full_img_path, w=180)
                pdf.ln(5)
            else:
                pdf.write(5, f"[Image not found: {img_path}]")
                pdf.ln(5)
        elif line.startswith('```'):
            continue
        elif line.startswith('* **') or line.startswith('- **') or line.startswith('1. ') or line.startswith('2. ') or line.startswith('3. '):
            try:
                pdf.multi_cell(180, 6, line)
            except Exception as e:
                continue
        elif line.startswith('|'):
            # simple table row print
            try:
                pdf.multi_cell(180, 6, line)
            except:
                continue
        else:
            try:
                pdf.multi_cell(180, 6, line)
            except Exception as e:
                print(f"Skipping line due to error: {e}")
                continue
    
    pdf.output(pdf_path)
    print(f"✅ PDF saved: {pdf_path}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python md_to_pdf.py <input.md> <output.pdf>")
        sys.exit(1)
    convert_md_to_pdf(sys.argv[1], sys.argv[2])
