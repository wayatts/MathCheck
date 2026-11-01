import os
import glob
import csv
import xml.etree.ElementTree as ET
from PIL import Image, ImageDraw

RAW_DIR = "data/raw"
OUTPUT_IMG_DIR = "data/processed/images"
OUTPUT_CSV = "data/processed/metadata.csv"
MAX_SAMPLES = 4000

os.makedirs(OUTPUT_IMG_DIR, exist_ok=True)
os.makedirs("data/processed", exist_ok=True)

VALID_CHARS = set("0123456789abcdefghijklmnopqrstuvwxyz+-*/=^()[]")

def parse_inkml(file_path):
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()
        ns = root.tag.split("}")[0] + "}" if root.tag.startswith("{") else ""

        latex_str = ""
        for ann in root.findall(f"{ns}annotation"):
            if ann.attrib.get("type") in ["truth", "normalized"]:
                latex_str = ann.text.strip() if ann.text else ""
                break

        traces = []
        for trace in root.findall(f"{ns}trace"):
            trace_text = trace.text.strip() if trace.text else ""
            coords = []
            for point in trace_text.split(","):
                pts = point.strip().split()
                if len(pts) >= 2:
                    coords.append((float(pts[0]), float(pts[1])))
            if len(coords) > 1:
                traces.append(coords)
                
        return latex_str, traces
    except Exception:
        return None, None

def render_strokes_fast(traces, output_path, target_size=(128, 32)):
    all_x = [pt[0] for stroke in traces for pt in stroke]
    all_y = [pt[1] for stroke in traces for pt in stroke]
    
    if not all_x or not all_y:
        return False
        
    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)
    
    span_x = max(1.0, max_x - min_x)
    span_y = max(1.0, max_y - min_y)

    pad = 4
    avail_w = target_size[0] - (2 * pad)
    avail_h = target_size[1] - (2 * pad)
    scale = min(avail_w / span_x, avail_h / span_y)

    img = Image.new("L", target_size, color=255)
    draw = ImageDraw.Draw(img)

    offset_x = pad + (avail_w - span_x * scale) / 2
    offset_y = pad + (avail_h - span_y * scale) / 2

    for stroke in traces:
        scaled_pts = [
            (int(offset_x + (pt[0] - min_x) * scale),
             int(offset_y + (pt[1] - min_y) * scale))
            for pt in stroke
        ]
        draw.line(scaled_pts, fill=0, width=2)

    img.save(output_path)
    return True

def clean_label(raw):
    """Normalizes LaTeX tokens to clean algebraic notation."""
    s = raw.replace("$", "").strip().lower()
    s = s.replace("\\times", "*").replace("\\cdot", "*").replace("\\div", "/")
    s = s.replace("\\pm", "+").replace("\\neq", "=").replace("\\leq", "<").replace("\\geq", ">")
    s = s.replace("\\left", "").replace("\\right", "")
    s = s.replace("{", "(").replace("}", ")")
    s = s.replace("\\frac", "")
    s = s.replace("\\", "")
    s = s.replace(" ", "")
    return s

def run_prep():
    inkml_files = glob.glob(os.path.join(RAW_DIR, "**", "*.inkml"), recursive=True)
    print(f"Discovered {len(inkml_files)} raw handwritten .inkml files.")

    count = 0
    with open(OUTPUT_CSV, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "label"])

        for fpath in inkml_files:
            raw_label, traces = parse_inkml(fpath)
            if not raw_label or not traces:
                continue

            label = clean_label(raw_label)
            

            if 2 <= len(label) <= 16 and all(c in VALID_CHARS for c in label):
                img_path = os.path.join(OUTPUT_IMG_DIR, f"hw_{count:05d}.png")
                success = render_strokes_fast(traces, img_path)
                if success:
                    writer.writerow([img_path, label])
                    count += 1
                    
                if count % 500 == 0:
                    print(f"Rendered {count}/{MAX_SAMPLES} samples...")

                if count >= MAX_SAMPLES:
                    break

    print(f"Successfully processed {count} real handwritten expressions into {OUTPUT_CSV}.")

if __name__ == "__main__":
    run_prep()