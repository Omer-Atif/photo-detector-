# Person In/Out of Car Detector 

Detects whether people in a photo are **inside** or **outside** a car.

## How it works

1. A pretrained YOLOv8 object-detection model finds every **car / bus /
   truck** in the image.
2. A pretrained YOLOv8 **pose** model finds every **person** along with
   their body keypoints (shoulders, hips, knees, ankles, etc.).
3. For each person who overlaps a vehicle's bounding box, the code looks
   at leg **posture**, not just visibility:
   - **Knee at roughly the same height as the hip** (bent leg, thigh
     close to horizontal) → sitting → **INSIDE** the car. This is what a
     driver/passenger looks like even with the door open and legs fully
     visible.
   - **Knee well below the hip** (leg extended, close to straight) →
     standing → **OUTSIDE** the car.
   - **Hip/knee not detected at all** (fully hidden behind the car body)
     → **INSIDE** the car.
   - People who don't overlap any vehicle at all are simply outside
     (unrelated to any car in the photo).
   - People cut off by the edge of the photo, where missing leg keypoints
     could just mean "out of frame" rather than hidden by the car, are
     treated as inconclusive and default to outside.
4. **Result labels**: each detected person gets a solid-background
   "INSIDE CAR" (green) or "OUTSIDE CAR" (red) tag drawn directly on the
   photo, sized to scale with the image, so it stays readable on both
   small and large photos.

**Why posture, not just box overlap or leg visibility?** Two earlier
approaches were tried and both had real failure cases:
- Box overlap alone: fails when someone stands right next to a car — same
  overlap pattern as sitting inside.
- "Are legs visible at all": fails on the most common real photo of
  someone in a car — door open, seated, legs clearly visible and bent.
  Visible legs were wrongly read as "must be standing."

Checking the actual bend of the leg (sitting vs. standing posture) fixes
both cases.

This is still a heuristic built on general-purpose detectors, not a model
trained specifically for this task, so it can still be fooled by unusual
angles, very heavy occlusion, people sitting on top of / hanging out of a
car, or photos where the car itself isn't recognizable to the detector
(e.g. an extreme close-up of just a dashboard).

## Setup

```bash
pip install -r requirements.txt
```

(The first run will automatically download the small YOLOv8n weights file,
~6MB, from GitHub.)

## Usage

### Option 1: Web app (drag & drop, multiple photos at once)

```bash
python app.py
```

Then open **http://127.0.0.1:5000**. You can:
- **Drag and drop** one or more photos onto the upload area, or click it
  to browse and select multiple files at once
- Click "Analyze Photos" to process all of them in one go

Each photo gets its own result card: the annotated image (people boxed in
green = inside, red = outside) plus a text summary. Up to 20 photos per
batch.

### Option 2: Command line

```bash
python cli.py path/to/photo.jpg -o annotated_output.jpg
```

## Files

- `detector.py` — core detection + classification logic
- `app.py` — Flask web app with an upload form
- `cli.py` — command-line version
- `requirements.txt` — dependencies

## Limitations

- Works best on clear, front/side-angle photos with a visible car and person.
- Can be less accurate on convertibles, motorcycles/other vehicle types
  (not covered), extreme close-ups, or crowded scenes.
- It's a rule-based heuristic, not a learned classifier — you can tighten
  or loosen the thresholds in `detector.py` (`overlap_thresh`,
  `x_containment_thresh`, `feet_tolerance_frac`) to tune behavior for your
  own photos.

  hope u like it ;)
