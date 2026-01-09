"""Generate synthetic petroleum engineering document for E2E testing.

This script creates a realistic 2-3 page PDF with:
- Technical content about API standards
- Tables with pressure ratings
- Proper formatting and structure
"""

from datetime import datetime
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


def create_sample_petroleum_doc(output_path: Path) -> None:
    """Create a sample petroleum engineering document.

    Args:
        output_path: Path where PDF will be saved
    """
    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=letter,
        rightMargin=0.75 * inch,
        leftMargin=0.75 * inch,
        topMargin=0.75 * inch,
        bottomMargin=0.75 * inch,
    )

    # Container for the 'Flowable' objects
    story = []
    styles = getSampleStyleSheet()

    # Create custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=18,
        textColor=colors.HexColor('#1a1a1a'),
        spaceAfter=30,
        alignment=1,  # Center
    )

    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=14,
        textColor=colors.HexColor('#2c3e50'),
        spaceAfter=12,
        spaceBefore=12,
    )

    # Title
    story.append(Paragraph("API 6A Wellhead and Christmas Tree Equipment", title_style))
    story.append(Paragraph("Pressure Rating and Material Specification Guide", title_style))
    story.append(Spacer(1, 0.3 * inch))

    # Document info
    info_text = f"""
    <b>Document Number:</b> API-TEST-001<br/>
    <b>Revision:</b> Rev C<br/>
    <b>Date:</b> {datetime.now().strftime('%B %Y')}<br/>
    <b>Status:</b> Approved for Engineering Use
    """
    story.append(Paragraph(info_text, styles['Normal']))
    story.append(Spacer(1, 0.4 * inch))

    # Section 1: Introduction
    story.append(Paragraph("1. INTRODUCTION", heading_style))

    intro_text = """
    This document provides pressure rating specifications and material requirements for
    wellhead and christmas tree equipment in accordance with API Specification 6A.
    All equipment shall be designed, manufactured, and tested to meet the requirements
    for PSL 3 (Product Specification Level 3) as defined in API 6A latest edition.
    """
    story.append(Paragraph(intro_text, styles['Normal']))
    story.append(Spacer(1, 0.2 * inch))

    # Section 2: Pressure Ratings
    story.append(Paragraph("2. PRESSURE RATING SPECIFICATIONS", heading_style))

    rating_text = """
    The maximum allowable working pressure for API 6A gate valves varies by size and
    pressure class. The following table provides standard pressure ratings for common
    valve configurations used in wellhead operations.
    """
    story.append(Paragraph(rating_text, styles['Normal']))
    story.append(Spacer(1, 0.2 * inch))

    # Table 1: API 6A Gate Valve Pressure Ratings
    table1_data = [
        ['Valve Size', 'Pressure Rating', 'Working Pressure', 'Test Pressure', 'Temp Range'],
        ['1-inch', '5000 PSI', '5000 PSI', '7500 PSI', '-20°F to 250°F'],
        ['2-inch', '5000 PSI', '5000 PSI', '7500 PSI', '-20°F to 250°F'],
        ['3-inch', '5000 PSI', '5000 PSI', '7500 PSI', '-20°F to 250°F'],
        ['2-inch', '10000 PSI', '10000 PSI', '15000 PSI', '-20°F to 250°F'],
        ['3-inch', '10000 PSI', '10000 PSI', '15000 PSI', '-20°F to 250°F'],
    ]

    table1 = Table(table1_data, colWidths=[1.2 * inch, 1.3 * inch, 1.3 * inch, 1.2 * inch, 1.5 * inch])
    table1.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#34495e')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('FONTSIZE', (0, 1), (-1, -1), 9),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f0f0f0')]),
    ]))

    story.append(table1)
    story.append(Spacer(1, 0.3 * inch))

    # Section 3: Temperature Derating
    story.append(Paragraph("3. TEMPERATURE DERATING FACTORS", heading_style))

    derating_text = """
    For Class 1500 flanges operating above standard temperature conditions, the following
    derating factors shall be applied to the maximum allowable working pressure. These
    factors account for the reduction in material strength at elevated temperatures.
    """
    story.append(Paragraph(derating_text, styles['Normal']))
    story.append(Spacer(1, 0.2 * inch))

    # Table 2: Temperature Derating Factors
    table2_data = [
        ['Temperature', 'Class 1500 Factor', 'Class 2500 Factor', 'Notes'],
        ['250°F', '1.00', '1.00', 'Standard rating'],
        ['300°F', '0.98', '0.98', 'Minimal derating'],
        ['400°F', '0.95', '0.94', 'Moderate derating'],
        ['500°F', '0.90', '0.88', 'Significant derating'],
        ['600°F', '0.85', '0.82', 'Major derating required'],
        ['700°F', '0.79', '0.75', 'Consult engineering'],
    ]

    table2 = Table(table2_data, colWidths=[1.3 * inch, 1.6 * inch, 1.6 * inch, 2.0 * inch])
    table2.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#34495e')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('FONTSIZE', (0, 1), (-1, -1), 9),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f0f0f0')]),
    ]))

    story.append(table2)
    story.append(Spacer(1, 0.3 * inch))

    # Page break
    story.append(PageBreak())

    # Section 4: Material Specifications
    story.append(Paragraph("4. MATERIAL SPECIFICATIONS AND BURST PRESSURE", heading_style))

    material_text = """
    Material selection is critical for ensuring safe operation under design conditions.
    The following table compares burst pressure ratings for different pipe materials at
    3-inch nominal diameter. Burst pressure represents the theoretical failure point and
    is typically 3-4 times the maximum allowable working pressure.
    """
    story.append(Paragraph(material_text, styles['Normal']))
    story.append(Spacer(1, 0.2 * inch))

    # Table 3: Burst Pressure by Material
    table3_data = [
        ['Material Grade', 'Nominal Size', 'Wall Thickness', 'Burst Pressure', 'SMYS'],
        ['Carbon Steel Grade B', '3-inch', '0.300 in', '8,200 PSI', '35,000 PSI'],
        ['Stainless Steel 316', '3-inch', '0.300 in', '9,100 PSI', '30,000 PSI'],
        ['Chrome 13% (F6NM)', '3-inch', '0.300 in', '9,500 PSI', '75,000 PSI'],
        ['Duplex 2205', '3-inch', '0.300 in', '10,800 PSI', '65,000 PSI'],
    ]

    table3 = Table(table3_data, colWidths=[1.8 * inch, 1.2 * inch, 1.3 * inch, 1.3 * inch, 1.2 * inch])
    table3.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#34495e')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 9),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('FONTSIZE', (0, 1), (-1, -1), 8),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f0f0f0')]),
    ]))

    story.append(table3)
    story.append(Spacer(1, 0.3 * inch))

    # Comparison note
    comparison_text = """
    <b>Material Comparison Analysis:</b> For 3-inch nominal diameter pipes, Stainless Steel 316
    provides approximately 11% higher burst strength compared to Carbon Steel Grade B
    (9,100 PSI vs 8,200 PSI). However, the higher cost of stainless steel must be justified
    by corrosion resistance requirements or specific service conditions such as sour gas
    environments.
    """
    story.append(Paragraph(comparison_text, styles['Normal']))
    story.append(Spacer(1, 0.3 * inch))

    # Section 5: Safety Requirements
    story.append(Paragraph("5. SAFETY AND OPERATIONAL REQUIREMENTS", heading_style))

    safety_text = """
    <b>5.1 H2S Service Requirements</b><br/>
    For sour gas service with H2S present, material selection must comply with NACE MR0175/ISO 15156.
    When H2S partial pressure exceeds 0.05 psi, only materials resistant to sulfide stress
    cracking (SSC) shall be used. Carbon steel is limited to maximum hardness of HRC 22.
    Grade 316/316L stainless steel or higher alloy materials are recommended for severe sour service.
    <br/><br/>
    <b>5.2 Emergency Shutdown Systems</b><br/>
    According to API RP 14C, emergency shutdown (ESD) valves on wellhead platforms must achieve
    full closure within 30 seconds maximum. All ESD valves shall be equipped with fail-safe
    spring return mechanisms to ensure closure upon loss of hydraulic or pneumatic pressure.
    <br/><br/>
    <b>5.3 Pressure Relief Requirements</b><br/>
    All pressure vessels and piping systems shall be protected by properly sized pressure
    relief valves. Relief valves must be set at or below the maximum allowable working pressure
    and shall be capable of preventing system pressure from exceeding 110% of design pressure
    during relief conditions.
    """
    story.append(Paragraph(safety_text, styles['Normal']))
    story.append(Spacer(1, 0.3 * inch))

    # Section 6: Inspection and Maintenance
    story.append(Paragraph("6. INSPECTION AND MAINTENANCE", heading_style))

    inspection_text = """
    <b>6.1 Corrosion Monitoring</b><br/>
    Visual inspection for corrosion indicators should be performed quarterly. Orange discoloration
    with pitting on valve exterior surfaces indicates active corrosion. If detected, immediately
    tag equipment out of service and measure pit depth using ultrasonic gauge. Pits exceeding
    10% of wall thickness require valve replacement. Lesser pitting should be evaluated per
    API 579 Fitness-For-Service guidelines.
    <br/><br/>
    <b>6.2 Pressure Relief Valve Testing</b><br/>
    Pressure relief valves experiencing chattering (rapid opening/closing cycles with audible
    clicking or rattling) indicate oversizing or backpressure issues. Common causes include
    set pressure too close to operating pressure, discharge backpressure exceeding 10% of
    set pressure, or excessive inlet pressure drop. Verify valve sizing calculations and
    inspect discharge piping for restrictions.
    <br/><br/>
    <b>6.3 Annular Pressure Monitoring</b><br/>
    Wellheads shall be monitored for annular pressure buildup (APB). Signs include sustained
    pressure on A or B annulus, pressure increasing over time between bleed-downs, or pressure
    exceeding hydrostatic expectations. Install permanent pressure gauges on all annuli and
    establish monitoring procedures per API RP 90.
    """
    story.append(Paragraph(inspection_text, styles['Normal']))
    story.append(Spacer(1, 0.2 * inch))

    # Footer
    story.append(Spacer(1, 0.5 * inch))
    footer_text = """
    <b>Note:</b> This document is intended for testing purposes only and contains simplified
    petroleum engineering content. For actual engineering applications, consult official
    API standards and qualified petroleum engineers.
    """
    story.append(Paragraph(footer_text, styles['Italic']))

    # Build PDF
    doc.build(story)
    print(f"✓ Generated sample petroleum document: {output_path}")


if __name__ == "__main__":
    output_path = Path(__file__).parent / "sample_petroleum_doc.pdf"
    create_sample_petroleum_doc(output_path)
