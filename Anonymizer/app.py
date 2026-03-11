import os
import tempfile
import json
import re
import traceback

from flask import Flask, request, jsonify, render_template, send_file
import fitz  # PyMuPDF
from transformers import pipeline

# ---------------------------------------------------------------------------
# Flask App
# ---------------------------------------------------------------------------
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB upload limit

# ---------------------------------------------------------------------------
# Load NER model once at startup
# ---------------------------------------------------------------------------
print("Loading NER model – this may take a moment on first run …")
ner_pipeline = pipeline(
    "ner",
    model="dslim/bert-base-NER",
    aggregation_strategy="simple",
)
print("NER model loaded ✓")

# Entity-label → placeholder mapping
LABEL_TO_PLACEHOLDER = {
    "PER": "[NAME]",
    "LOC": "[ADDRESS]",
    "ORG": "[ORGANIZATION]",
}

# Regex patterns
EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}\b")
PHONE_RE = re.compile(
    r"\b\d{10}\b|\(\d{3}\)\s*\d{3}-\d{4}|\d{3}-\d{3}-\d{4}|\+\d{1,3}\s?\d{10}"
)
AGE_RE = re.compile(r"\b\d{1,3}\s*(years|year|yrs|yr)\b", re.IGNORECASE)
DATE_RE = re.compile(
    r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b"
)

# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def extract_text_from_pdf(file_path: str) -> list[str]:
    """Return a list of strings, one per page."""
    doc = fitz.open(file_path)
    pages = []
    for page in doc:
        pages.append(page.get_text("text"))
    doc.close()
    return pages


def anonymize_with_ner(text: str) -> str:
    """Replace NER-detected entities with placeholders."""
    entities = ner_pipeline(text)
    # Sort by start position descending so replacements don't shift indices
    entities.sort(key=lambda e: e["start"], reverse=True)
    for ent in entities:
        label = ent["entity_group"]
        if label in LABEL_TO_PLACEHOLDER:
            placeholder = LABEL_TO_PLACEHOLDER[label]
            text = text[:ent["start"]] + placeholder + text[ent["end"]:]
    return text


def anonymize_with_regex(text: str) -> str:
    """Replace emails, phones, ages, and dates with placeholders."""
    text = EMAIL_RE.sub("[EMAIL]", text)
    text = PHONE_RE.sub("[PHONE]", text)
    text = AGE_RE.sub("[AGE]", text)
    text = DATE_RE.sub("[DATE]", text)
    return text


def process_page(text: str) -> dict:
    """Anonymize a single page and return structured data."""
    anonymized = anonymize_with_ner(text)
    anonymized = anonymize_with_regex(anonymized)

    # Split into non-empty lines
    lines = [ln.strip() for ln in anonymized.splitlines() if ln.strip()]

    # Build simple sections by detecting "header-like" lines
    sections: list[dict] = []
    current_section = {"heading": "General", "lines": []}

    for line in lines:
        # Heuristic: all-caps line or line ending with ':' is a heading
        if (line.isupper() and len(line) > 2) or (line.endswith(":") and len(line) < 60):
            if current_section["lines"]:
                sections.append(current_section)
            current_section = {"heading": line.rstrip(":"), "lines": []}
        else:
            current_section["lines"].append(line)

    if current_section["lines"] or not sections:
        sections.append(current_section)

    return {"sections": sections}


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    """Accept a PDF, anonymize it, return structured JSON."""
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    if not file.filename.lower().endswith(".pdf"):
        return jsonify({"error": "Only PDF files are accepted"}), 400

    try:
        # Save to temp file
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        file.save(tmp.name)
        tmp.close()

        # Extract & anonymize
        pages_text = extract_text_from_pdf(tmp.name)
        pages_data = [process_page(page) for page in pages_text]

        return jsonify({"pages": pages_data, "total_pages": len(pages_data)})

    except Exception as exc:
        traceback.print_exc()
        return jsonify({"error": f"Processing failed: {str(exc)}"}), 500

    finally:
        # Clean up temp file
        try:
            os.unlink(tmp.name)
        except Exception:
            pass


@app.route("/download-pdf", methods=["POST"])
def download_pdf():
    """Accept structured JSON, generate a well-formatted PDF, return it."""
    data = request.get_json(silent=True)
    if not data or "pages" not in data:
        return jsonify({"error": "No page data provided"}), 400

    try:
        doc = fitz.open()  # new empty PDF

        for page_data in data["pages"]:
            page = doc.new_page(width=595, height=842)  # A4
            y = 50

            for section in page_data.get("sections", []):
                heading = section.get("heading", "")
                lines = section.get("lines", [])

                # Draw heading
                if heading:
                    page.insert_text(
                        fitz.Point(50, y),
                        heading.upper(),
                        fontsize=13,
                        fontname="helv",
                        color=(0.15, 0.15, 0.15),
                    )
                    y += 20
                    # Draw underline
                    page.draw_line(
                        fitz.Point(50, y - 4), fitz.Point(545, y - 4),
                        color=(0.7, 0.7, 0.7), width=0.5
                    )
                    y += 8

                # Draw lines
                for line in lines:
                    if y > 790:  # new page if overflow
                        page = doc.new_page(width=595, height=842)
                        y = 50

                    page.insert_text(
                        fitz.Point(55, y),
                        line,
                        fontsize=10,
                        fontname="helv",
                        color=(0.2, 0.2, 0.2),
                    )
                    y += 15

                y += 10  # spacing between sections

        # Save to temp file & send
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        doc.save(tmp.name)
        doc.close()
        tmp.close()

        return send_file(
            tmp.name,
            as_attachment=True,
            download_name="anonymized_resume.pdf",
            mimetype="application/pdf",
        )

    except Exception as exc:
        traceback.print_exc()
        return jsonify({"error": f"PDF generation failed: {str(exc)}"}), 500


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True, port=5000)
