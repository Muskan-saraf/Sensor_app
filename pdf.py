from fpdf import FPDF
import os
import pandas as pd

TEMP_FOLDER = "temp"
STATIC_FOLDER = "static"
os.makedirs(STATIC_FOLDER, exist_ok=True)

def render_table_to_pdf(pdf, df, title, font_size=6, max_width=290):
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, title, ln=True)
    pdf.set_font("Arial", "", font_size)

    col_width = max_width / len(df.columns)

    # Header
    for col in df.columns:
        header = str(col)[:20]
        pdf.cell(col_width, 6, header, border=1, align="C")
    pdf.ln()

    # Rows
    for row in df.itertuples(index=False):
        for value in row:
            val = str(value)
            if len(val) > 15:
                val = val[:13] + "..."  # safe for fpdf (ASCII)
            pdf.cell(col_width, 6, val, border=1, align="C")
        pdf.ln()
    pdf.ln(5)

# def render_image_to_pdf(pdf, image_path, title):
   # if os.path.exists(image_path):
    #    pdf.add_page()
     #   pdf.set_font("Arial", "B", 14)
      #  pdf.cell(0, 10, title, ln=True)
       # pdf.ln(5)
        #pdf.image(image_path, x=10, w=pdf.w - 20)
       # pdf.ln(10)

def generate_pdf():
    # File paths
    stats_path = os.path.join(TEMP_FOLDER, "stats_table.pkl")
    health_path = os.path.join(TEMP_FOLDER, "results_df.pkl")
    stability_path = os.path.join(TEMP_FOLDER, "stability_df.pkl")

    # Validate presence
    if not os.path.exists(stats_path):
        return None, "Error: stats_table.pkl not found"

    pdf = FPDF(orientation='L', unit='mm', format='A4')

    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Main Title
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Sensor Data Analysis Report", ln=True, align="C")
    pdf.ln(5)

    # === 1. Statistics Table ===
    stats_table = pd.read_pickle(stats_path)
    render_table_to_pdf(pdf, stats_table, "Statistics Summary")

    # === 2. Sensor Health Table ===
    if os.path.exists(health_path):
        health_df = pd.read_pickle(health_path)
        render_table_to_pdf(pdf, health_df, "Sensor Health Check")

    # === 3. Feature Stability Table ===
    if os.path.exists(stability_path):
        feature_df = pd.read_pickle(stability_path)
        render_table_to_pdf(pdf, feature_df, "Feature Stability Analysis")

    # === 4. Stability Summary Table ===
    stability_result_path = os.path.join(TEMP_FOLDER, "stability_result_df.pkl")
    if os.path.exists(stability_result_path):
        model_stability_df = pd.read_pickle(stability_result_path)
        render_table_to_pdf(pdf, model_stability_df, "Model Stability Summary")


    # === 7. Plots ===
    # image_plots = [
        
     #   ("static/timeseries_stacked.png", "Time Series Plot"),
     #   ("static/anomalies_stacked.png", "Anomaly Detection Plot"),
     #   ("static/correlation_heatmap_large.png", "Correlation Heatmap")
    # ]

    # for image_path, title in image_plots:
     #    render_image_to_pdf(pdf, image_path, title)

    # Save PDF
    pdf_path = os.path.join(STATIC_FOLDER, "sensor_report.pdf")
    pdf.output(pdf_path)
    return pdf_path, None
