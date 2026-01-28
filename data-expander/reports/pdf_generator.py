import io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak, Table, TableStyle, PageTemplate, Frame
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.pdfgen import canvas
from datetime import datetime

# Professional color palette
BRAND_PRIMARY = colors.HexColor('#2C3E50')  # Dark blue-grey
BRAND_SECONDARY = colors.HexColor('#3498DB')  # Bright blue
BRAND_ACCENT = colors.HexColor('#E74C3C')  # Red
BRAND_SUCCESS = colors.HexColor('#27AE60')  # Green
BRAND_WARNING = colors.HexColor('#F39C12')  # Orange
BRAND_LIGHT = colors.HexColor('#ECF0F1')  # Light grey

# Set professional matplotlib style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class HeaderFooterCanvas(canvas.Canvas):
    """Custom canvas with branded header and footer on each page"""
    
    def __init__(self, *args, **kwargs):
        canvas.Canvas.__init__(self, *args, **kwargs)
        self.pages = []
        
    def showPage(self):
        self.pages.append(dict(self.__dict__))
        self._startPage()
        
    def save(self):
        page_count = len(self.pages)
        for page_num, page in enumerate(self.pages, start=1):
            self.__dict__.update(page)
            self.draw_header_footer(page_num, page_count)
            canvas.Canvas.showPage(self)
        canvas.Canvas.save(self)
        
    def draw_header_footer(self, page_num, page_count):
        # Header
        self.setStrokeColor(BRAND_SECONDARY)
        self.setLineWidth(2)
        self.line(0.75*inch, 10.5*inch, 7.75*inch, 10.5*inch)
        
        self.setFont('Helvetica-Bold', 10)
        self.setFillColor(BRAND_PRIMARY)
        self.drawString(0.75*inch, 10.65*inch, "Data Expander Pro")
        
        self.setFont('Helvetica', 8)
        self.setFillColor(colors.grey)
        self.drawRightString(7.75*inch, 10.65*inch, f"Data Quality Assessment Report")
        
        # Footer
        self.setStrokeColor(BRAND_LIGHT)
        self.setLineWidth(1)
        self.line(0.75*inch, 0.6*inch, 7.75*inch, 0.6*inch)
        
        self.setFont('Helvetica', 8)
        self.setFillColor(colors.grey)
        self.drawString(0.75*inch, 0.4*inch, f"Generated: {datetime.now().strftime('%B %d, %Y')}")
        self.drawCentredString(4.25*inch, 0.4*inch, "Confidential - For Internal Use Only")
        self.drawRightString(7.75*inch, 0.4*inch, f"Page {page_num} of {page_count}")


def generate_professional_pdf_report(df, dataset_name, health_data, findings=None):
    """
    Generates an industry-standard professional PDF report.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer, 
        pagesize=letter,
        rightMargin=0.75*inch, 
        leftMargin=0.75*inch,
        topMargin=1.2*inch, 
        bottomMargin=0.9*inch
    )
    
    story = []
    styles = getSampleStyleSheet()
    
    # Custom professional styles
    title_style = ParagraphStyle(
        'ExecutiveTitle',
        parent=styles['Heading1'],
        fontSize=32,
        textColor=BRAND_PRIMARY,
        spaceAfter=0.2*inch,
        alignment=TA_CENTER,
        fontName='Helvetica-Bold',
        leading=38
    )
    
    subtitle_style = ParagraphStyle(
        'Subtitle',
        parent=styles['Normal'],
        fontSize=14,
        textColor=BRAND_SECONDARY,
        spaceAfter=0.5*inch,
        alignment=TA_CENTER,
        fontName='Helvetica'
    )
    
    heading_style = ParagraphStyle(
        'SectionHeading',
        parent=styles['Heading2'],
        fontSize=18,
        textColor=BRAND_PRIMARY,
        spaceAfter=0.15*inch,
        spaceBefore=0.2*inch,
        fontName='Helvetica-Bold'
    )
    
    body_style = ParagraphStyle(
        'Body',
        parent=styles['Normal'],
        fontSize=11,
        textColor=colors.HexColor('#2C3E50'),
        alignment=TA_JUSTIFY,
        leading=16,
        spaceAfter=8
    )
    
    # ===== COVER PAGE =====
    story.append(Spacer(1, 1.5*inch))
    story.append(Paragraph("DATA QUALITY", title_style))
    story.append(Paragraph("ASSESSMENT REPORT", title_style))
    story.append(Spacer(1, 0.1*inch))
    story.append(Paragraph(f"{dataset_name}", subtitle_style))
    
    # Grade badge
    grade_color = BRAND_SUCCESS if health_data['grade'] in ['A', 'B'] else BRAND_WARNING if health_data['grade'] == 'C' else BRAND_ACCENT
    
    grade_data = [[
        Paragraph(f"<font size=48 color={grade_color.hexval()}><b>{health_data['grade']}</b></font>", styles['Normal']),
        Paragraph(f"<font size=14><b>Overall Grade</b></font><br/><font size=11>Health Score: {health_data['score']}/100</font>", body_style)
    ]]
    
    grade_table = Table(grade_data, colWidths=[1.2*inch, 3*inch])
    grade_table.setStyle(TableStyle([
        ('ALIGN', (0, 0), (0, 0), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('BOX', (0, 0), (-1, -1), 2, grade_color),
        ('BACKGROUND', (0, 0), (0, 0), BRAND_LIGHT),
        ('LEFTPADDING', (0, 0), (-1, -1), 15),
        ('RIGHTPADDING', (0, 0), (-1, -1), 15),
        ('TOPPADDING', (0, 0), (-1, -1), 15),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 15),
    ]))
    
    story.append(Spacer(1, 0.5*inch))
    story.append(grade_table)
    story.append(Spacer(1, 0.5*inch))
    
    # Key metrics table
    metrics_data = [
        ['Dataset Information', ''],
        ['Total Records:', f"{len(df):,}"],
        ['Total Features:', f"{len(df.columns)}"],
        ['Missing Data:', f"{health_data['missing_ratio']:.2%}"],
        ['Duplicate Records:', f"{health_data['duplicate_ratio']:.2%}"],
        ['Report Date:', datetime.now().strftime("%B %d, %Y at %I:%M %p")],
    ]
    
    metrics_table = Table(metrics_data, colWidths=[2.5*inch, 2.5*inch])
    metrics_table.setStyle(TableStyle([
        ('FONT', (0, 0), (-1, 0), 'Helvetica-Bold', 12),
        ('FONT', (0, 1), (0, -1), 'Helvetica-Bold', 10),
        ('FONT', (1, 1), (1, -1), 'Helvetica', 10),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('BACKGROUND', (0, 0), (-1, 0), BRAND_PRIMARY),
        ('TEXTCOLOR', (0, 1), (-1, -1), BRAND_PRIMARY),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, BRAND_LIGHT]),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('LEFTPADDING', (0, 0), (-1, -1), 12),
        ('RIGHTPADDING', (0, 0), (-1, -1), 12),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
    ]))
    
    story.append(metrics_table)
    story.append(PageBreak())
    
    # ===== EXECUTIVE SUMMARY =====
    story.append(Paragraph("EXECUTIVE SUMMARY", heading_style))
    story.append(Spacer(1, 0.1*inch))
    
    summary_text = f"""
    This comprehensive assessment evaluated <b>{dataset_name}</b> across multiple dimensions of data quality, 
    integrity, and ML-readiness. The dataset contains <b>{len(df):,} records</b> and <b>{len(df.columns)} features</b>.
    <br/><br/>
    <b>Overall Health Score: {health_data['score']}/100 (Grade {health_data['grade']})</b><br/>
    {get_grade_interpretation(health_data['grade'])}
    <br/><br/>
    <b>Critical Findings:</b><br/>
    • Data Completeness: {(1-health_data['missing_ratio'])*100:.1f}% complete<br/>
    • Record Uniqueness: {(1-health_data['duplicate_ratio'])*100:.1f}% unique<br/>
    • Quality Deductions: {sum(health_data['deductions'].values())} points across {len(health_data['deductions'])} categories<br/>
    <br/>
    <b>Recommendation:</b> <i>{get_recommendation(health_data['score'])}</i>
    """
    
    story.append(Paragraph(summary_text, body_style))
    story.append(Spacer(1, 0.3*inch))
    
    # ===== DATA QUALITY VISUALIZATIONS =====
    story.append(Paragraph("DATA QUALITY ANALYSIS", heading_style))
    story.append(Spacer(1, 0.15*inch))
    
    # Chart 1: Missing data visualization
    fig, ax = plt.subplots(figsize=(7, 3.5))
    missing_data = df.isnull().sum()
    
    if missing_data.sum() > 0:
        top_missing = missing_data[missing_data > 0].head(10)
        colors_bar = ['#E74C3C' if x > len(df)*0.1 else '#F39C12' if x > len(df)*0.05 else '#3498DB' for x in top_missing.values]
        
        ax.barh(range(len(top_missing)), top_missing.values, color=colors_bar, edgecolor='white', linewidth=1.5)
        ax.set_yticks(range(len(top_missing)))
        ax.set_yticklabels(top_missing.index, fontsize=9)
        ax.set_xlabel('Missing Values Count', fontsize=10, fontweight='bold')
        ax.set_title('Missing Data Distribution (Top 10 Features)', fontsize=12, fontweight='bold', pad=15)
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    else:
        ax.text(0.5, 0.5, '✓ No Missing Data Detected', 
                ha='center', va='center', fontsize=18, color='#27AE60', fontweight='bold')
        ax.axis('off')
    
    fig.patch.set_facecolor('white')
    ax.set_facecolor('#FAFAFA')
    
    img_buffer = io.BytesIO()
    plt.tight_layout()
    plt.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight', facecolor='white')
    img_buffer.seek(0)
    plt.close()
    
    story.append(Image(img_buffer, width=6*inch, height=3*inch))
    story.append(Spacer(1, 0.3*inch))
    
    # Chart 2: Data type pie chart
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3))
    
    dtype_counts = df.dtypes.value_counts()
    colors_pie = ['#2C3E50', '#3498DB', '#27AE60', '#F39C12', '#E74C3C']
    
    wedges, texts, autotexts = ax1.pie(dtype_counts.values, 
                                        labels=[str(x) for x in dtype_counts.index], 
                                        autopct='%1.1f%%',
                                        colors=colors_pie[:len(dtype_counts)],
                                        startangle=90,
                                        textprops={'fontsize': 9, 'weight': 'bold'})
    
    for autotext in autotexts:
        autotext.set_color('white')
    
    ax1.set_title('Feature Types Distribution', fontsize=11, fontweight='bold', pad=10)
    
    # Completeness gauge
    completeness = 1 - health_data['missing_ratio']
    gauge_colors = ['#27AE60' if completeness > 0.95 else '#F39C12' if completeness > 0.8 else '#E74C3C', '#E0E0E0']
    
    ax2.pie([completeness, 1-completeness], 
            colors=gauge_colors,
            startangle=90,
            counterclock=False,
            wedgeprops={'width': 0.3})
    
    ax2.text(0, 0, f'{completeness*100:.1f}%', ha='center', va='center', fontsize=24, fontweight='bold', color=gauge_colors[0])
    ax2.set_title('Data Completeness', fontsize=11, fontweight='bold', pad=10)
    
    fig.patch.set_facecolor('white')
    
    img_buffer2 = io.BytesIO()
    plt.tight_layout()
    plt.savefig(img_buffer2, format='png', dpi=300, bbox_inches='tight', facecolor='white')
    img_buffer2.seek(0)
    plt.close()
    
    story.append(Image(img_buffer2, width=6*inch, height=2.5*inch))
    story.append(PageBreak())
    
    # ===== DETAILED FINDINGS =====
    story.append(Paragraph("DETAILED ASSESSMENT", heading_style))
    story.append(Spacer(1, 0.15*inch))
    
    # Deductions table
    if health_data['deductions']:
        story.append(Paragraph("<b>Quality Deductions Breakdown:</b>", body_style))
        story.append(Spacer(1, 0.1*inch))
        
        deduction_data = [['Issue Category', 'Points Deducted', 'Impact']]
        for issue, points in health_data['deductions'].items():
            if points > 0:
                impact = 'High' if points >= 15 else 'Medium' if points >= 8 else 'Low'
                deduction_data.append([issue, f"-{points}", impact])
        
        deduction_table = Table(deduction_data, colWidths=[3*inch, 1.2*inch, 1*inch])
        deduction_table.setStyle(TableStyle([
            ('FONT', (0, 0), (-1, 0), 'Helvetica-Bold', 10),
            ('FONT', (0, 1), (-1, -1), 'Helvetica', 9),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('BACKGROUND', (0, 0), (-1, 0), BRAND_PRIMARY),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, BRAND_LIGHT]),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('ALIGN', (1, 0), (2, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('LEFTPADDING', (0, 0), (-1, -1), 10),
            ('RIGHTPADDING', (0, 0), (-1, -1), 10),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ]))
        
        story.append(deduction_table)
        story.append(Spacer(1, 0.3*inch))
    
    # Additional findings
    if findings:
        story.append(Paragraph("<b>Additional Findings:</b>", body_style))
        story.append(Spacer(1, 0.1*inch))
        
        if 'pii' in findings and findings['pii']:
            pii_text = f"<font color='{BRAND_ACCENT.hexval()}'><b>⚠ Privacy Alert:</b></font> PII detected in {len(findings['pii'])} column(s): {', '.join(list(findings['pii'].keys())[:5])}"
            story.append(Paragraph(pii_text, body_style))
        
        if 'fairness' in findings and findings['fairness']:
            story.append(Paragraph(f"<font color='{BRAND_WARNING.hexval()}'><b>⚖ Fairness Issue:</b></font> {findings['fairness']}", body_style))
        
        if 'leakage' in findings and findings['leakage']:
            story.append(Paragraph(f"<font color='{BRAND_ACCENT.hexval()}'><b>🚨 Data Leakage Risk:</b></font> {findings['leakage']}", body_style))
    
    story.append(Spacer(1, 0.4*inch))
    
    # ===== RECOMMENDATIONS =====
    story.append(Paragraph("ACTIONABLE RECOMMENDATIONS", heading_style))
    story.append(Spacer(1, 0.15*inch))
    
    recommendations = generate_recommendations(df, health_data, findings)
    
    for i, rec in enumerate(recommendations, 1):
        priority = "🔴 HIGH" if i <= 2 else "🟡 MEDIUM" if i <= 4 else "🟢 LOW"
        rec_para = Paragraph(f"<b>{i}. [{priority}]</b> {rec}", body_style)
        story.append(rec_para)
        story.append(Spacer(1, 0.08*inch))
    
    # ===== CONCLUSION =====
    story.append(Spacer(1, 0.3*inch))
    story.append(Paragraph("CONCLUSION", heading_style))
    conclusion_text = get_conclusion(health_data['score'], len(df))
    story.append(Paragraph(conclusion_text, body_style))
    
    # Build PDF with custom canvas
    doc.build(story, canvasmaker=HeaderFooterCanvas)
    buffer.seek(0)
    return buffer


def get_grade_interpretation(grade):
    interpretations = {
        'A': 'Outstanding data quality. This dataset meets enterprise production standards.',
        'B': 'Good data quality with minor areas for improvement. Suitable for most ML applications.',
        'C': 'Moderate data quality. Remediation recommended before critical applications.',
        'D': 'Below-average quality. Significant cleaning required.',
        'F': 'Poor data quality. Extensive remediation mandatory before use.'
    }
    return interpretations.get(grade, 'Quality assessment unavailable.')


def get_recommendation(score):
    if score >= 90:
        return "This dataset is production-ready and meets enterprise quality standards."
    elif score >= 75:
        return "Address the flagged issues, then proceed with confidence to model development."
    elif score >= 50:
        return "Significant data cleaning is recommended. Use the Auto-Remediation feature before ML training."
    else:
        return "Critical quality issues detected. Extensive remediation required using ML-Safe cleaning workflow."


def get_conclusion(score, rows):
    if score >= 85:
        return f"This {rows:,}-record dataset demonstrates excellent data quality and is ready for immediate deployment in production ML pipelines. The minimal issues identified can be addressed through standard preprocessing workflows."
    elif score >= 70:
        return f"With {rows:,} records, this dataset shows promise but requires targeted improvements in the areas highlighted above. Following the recommendations will elevate this to production-grade quality."
    else:
        return f"While this {rows:,}-record dataset contains valuable information, significant quality improvements are necessary before deployment. Prioritize the high-impact recommendations and utilize the ML-Safe cleaning features to ensure proper data hygiene."


def generate_recommendations(df, health_data, findings):
    recs = []
    
    if health_data['missing_ratio'] > 0.05:
        recs.append("Implement ML-Safe imputation using the IterativeImputer to handle missing values without introducing data leakage.")
    
    if health_data['duplicate_ratio'] > 0.01:
        recs.append(f"Remove {int(len(df)*health_data['duplicate_ratio'])} duplicate records to improve model efficiency and reduce overfitting risk.")
    
    if findings and 'pii' in findings and findings['pii']:
        recs.append("Apply PII anonymization or removal before sharing this dataset externally or deploying models to production.")
    
    if findings and 'leakage' in findings and findings['leakage']:
        recs.append("Conduct Temporal Split Simulation to verify that no future information is leaking into your feature set.")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        skewed_cols = [col for col in numeric_cols if abs(df[col].skew()) > 2]
        if skewed_cols:
            recs.append(f"Apply log or Box-Cox transformation to highly skewed features: {', '.join(skewed_cols[:3])}{'...' if len(skewed_cols) > 3 else ''}")
    
    if health_data['score'] < 60:
        recs.append("Use the ML-Safe Split-Aware Cleaning feature to prevent test set leakage during remediation.")
    
    if len(recs) == 0:
        recs.append("Continue with your current data quality practices. This dataset exceeds industry standards.")
    
    return recs
