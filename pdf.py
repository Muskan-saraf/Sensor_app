from fpdf import FPDF
import os
import pandas as pd
from flask import send_file

TEMP_FOLDER = "temp"
STATIC_FOLDER = "static"
os.makedirs(STATIC_FOLDER, exist_ok=True)

def render_table_to_pdf(pdf, df, title, max_width=190):
    pdf.set_font("Arial", "B", 12)
    pdf.cell(200, 10, title, ln=True)
    pdf.set_font("Arial", "", 8)
    col_width = max_width / len(df.columns)

    # Header
    for col in df.columns:
        pdf.cell(col_width, 8, str(col)[:20], border=1, align="C")
    pdf.ln()

    # Rows
    for row in df.itertuples(index=False):
        for value in row:
            value_str = str(value)
            if len(value_str) > 30:
                value_str = value_str[:27] + "..."
            pdf.cell(col_width, 8, value_str, border=1, align="C")
        pdf.ln()
    pdf.ln(10)

def generate_pdf():
    stats_path = os.path.join(TEMP_FOLDER, "stats_table.pkl")
    if not os.path.exists(stats_path):
        return None, "Error: stats_table.pkl not found"

    stats_table = pd.read_pickle(stats_path)

    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    pdf.set_font("Arial", "B", 16)
    pdf.cell(200, 10, "Sensor Data Statistics Report", ln=True, align="C")
    pdf.ln(10)

    render_table_to_pdf(pdf, stats_table, "Statistics Summary")

    pdf_path = os.path.join(STATIC_FOLDER, "sensor_report.pdf")
    pdf.output(pdf_path)

    return pdf_path, None
