"""
Flask web app: drag-and-drop (or click to browse) one or more photos, and
it tells you whether each detected person is inside or outside a car.

Run with:
    python app.py
Then open http://127.0.0.1:5000 in your browser.
"""

import os
import uuid
from flask import Flask, request, render_template_string
from werkzeug.utils import secure_filename
from detector import analyze_image

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

MAX_FILES_PER_REQUEST = 20

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024  # 100MB total per request

PAGE = """
<!doctype html>
<html>
<head>
    <title>Person In/Out of Car Detector</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 900px; margin: 40px auto; padding: 0 20px; background: #f8fafc; }
        h1 { font-size: 1.5em; }

        #dropzone {
            border: 3px dashed #999;
            border-radius: 10px;
            padding: 40px 20px;
            text-align: center;
            cursor: pointer;
            transition: background 0.15s, border-color 0.15s;
            background: white;
        }
        #dropzone.dragover { background: #eef6ff; border-color: #3b82f6; }
        #dropzone p { margin: 8px 0; color: #555; }
        #fileInput { display: none; }

        #fileList { margin-top: 14px; text-align: left; font-size: 0.9em; color: #333; }
        #fileList div { padding: 2px 0; }

        #submitBtn {
            margin-top: 16px; padding: 10px 22px; font-size: 1em;
            border: none; border-radius: 6px; background: #2563eb; color: white;
            cursor: pointer;
        }
        #submitBtn:disabled { background: #9ca3af; cursor: not-allowed; }

        .result-card {
            border: 1px solid #e5e7eb; border-radius: 10px; padding: 18px 20px;
            margin-top: 18px; background: white; box-shadow: 0 1px 3px rgba(0,0,0,0.04);
        }
        .result-card h3 { margin-top: 0; margin-bottom: 12px; color: #111827; font-size: 1.05em; }

        .badge-row { display: flex; flex-direction: column; gap: 10px; }
        .badge {
            display: flex; align-items: center; gap: 12px;
            padding: 14px 18px; border-radius: 10px; font-size: 1.05em; font-weight: 600;
        }
        .badge .icon {
            display: flex; align-items: center; justify-content: center;
            width: 34px; height: 34px; border-radius: 50%; font-size: 1.1em; flex-shrink: 0;
        }
        .badge .sub { font-weight: 400; font-size: 0.85em; opacity: 0.8; margin-left: auto; }

        .badge.inside { background: #ecfdf5; color: #065f46; border: 1px solid #a7f3d0; }
        .badge.inside .icon { background: #10b981; color: white; }

        .badge.outside { background: #fef2f2; color: #991b1b; border: 1px solid #fecaca; }
        .badge.outside .icon { background: #ef4444; color: white; }

        .badge.none { background: #f3f4f6; color: #4b5563; border: 1px solid #e5e7eb; }
        .badge.none .icon { background: #9ca3af; color: white; }

        .error { color: #b91c1c; }
    </style>
</head>
<body>
    <h1>🚗 Is the Person In or Out of the Car?</h1>

    <form method="POST" enctype="multipart/form-data" id="uploadForm">
        <div id="dropzone">
            <p><strong>Drag & drop photos here</strong></p>
            <p>or click to browse (you can select multiple files)</p>
            <input type="file" id="fileInput" name="photos" accept="image/*" multiple>
            <div id="fileList"></div>
        </div>
        <div style="text-align:center;">
            <button type="submit" id="submitBtn" disabled>Analyze Photos</button>
        </div>
    </form>

    <script>
        const dropzone = document.getElementById('dropzone');
        const fileInput = document.getElementById('fileInput');
        const fileList = document.getElementById('fileList');
        const submitBtn = document.getElementById('submitBtn');

        function updateFileList(files) {
            fileList.innerHTML = '';
            if (files.length > 0) {
                for (const f of files) {
                    const d = document.createElement('div');
                    d.textContent = '📷 ' + f.name;
                    fileList.appendChild(d);
                }
                submitBtn.disabled = false;
            } else {
                submitBtn.disabled = true;
            }
        }

        dropzone.addEventListener('click', () => fileInput.click());

        fileInput.addEventListener('change', () => updateFileList(fileInput.files));

        ['dragenter', 'dragover'].forEach(evt =>
            dropzone.addEventListener(evt, (e) => {
                e.preventDefault();
                dropzone.classList.add('dragover');
            })
        );

        ['dragleave', 'drop'].forEach(evt =>
            dropzone.addEventListener(evt, (e) => {
                e.preventDefault();
                dropzone.classList.remove('dragover');
            })
        );

        dropzone.addEventListener('drop', (e) => {
            const dt = new DataTransfer();
            for (const f of e.dataTransfer.files) {
                if (f.type.startsWith('image/')) dt.items.add(f);
            }
            fileInput.files = dt.files;
            updateFileList(fileInput.files);
        });

        document.getElementById('uploadForm').addEventListener('submit', () => {
            submitBtn.disabled = true;
            submitBtn.textContent = 'Analyzing...';
        });
    </script>

    {% if results %}
        <h2>Results</h2>

        {% for r in results %}
            <div class="result-card">
                <h3>{{ r.name }}</h3>
                {% if r.error %}
                    <p class="error">Couldn't process this file: {{ r.error }}</p>
                {% else %}
                    <div class="badge-row">
                        {% for v in r.verdicts %}
                            <div class="badge {{ v.css }}">
                                <span class="icon">{{ v.icon }}</span>
                                <span>{{ v.label }}</span>
                                {% if r.verdicts|length > 1 %}
                                    <span class="sub">Person {{ loop.index }}</span>
                                {% endif %}
                            </div>
                        {% endfor %}
                    </div>
                {% endif %}
            </div>
        {% endfor %}
    {% endif %}
</body>
</html>
"""


def _verdicts_for(people):
    """Build the display data for each detected person's badge (or a single
    'no people detected' badge if none were found)."""
    if not people:
        return [{"css": "none", "icon": "?", "label": "No people detected"}]

    verdicts = []
    for pr in people:
        if pr.status == "inside":
            verdicts.append({"css": "inside", "icon": "✓", "label": "Inside the car"})
        else:
            verdicts.append({"css": "outside", "icon": "✕", "label": "Outside the car"})
    return verdicts


@app.route("/", methods=["GET", "POST"])
def index():
    results = []

    if request.method == "POST":
        files = request.files.getlist("photos")[:MAX_FILES_PER_REQUEST]

        for file in files:
            if not file or not file.filename:
                continue

            original_name = secure_filename(file.filename)
            unique_prefix = uuid.uuid4().hex[:8]
            upload_path = os.path.join(UPLOAD_DIR, f"{unique_prefix}_{original_name}")

            try:
                file.save(upload_path)
                analysis = analyze_image(upload_path)

                results.append({
                    "name": original_name,
                    "verdicts": _verdicts_for(analysis["people"]),
                    "error": None,
                })
            except Exception as exc:
                results.append({
                    "name": original_name,
                    "verdicts": None,
                    "error": str(exc),
                })

    return render_template_string(PAGE, results=results)


if __name__ == "__main__":
    app.run(debug=True, port=5000)