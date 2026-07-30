"""
Core logic: detect people and cars in an image, then decide whether each
person is INSIDE or OUTSIDE a car.

Why posture, not just "are legs visible"
------------------------------------------
Version 1 used bounding-box overlap alone -> failed when someone stood
right next to a car (box overlap looks identical to sitting inside).

Version 2 switched to "are the ankles/knees visible" -> failed on the most
common real photo of someone IN a car: door open, seated, legs clearly
visible and bent. Visible knees were being read as "must be standing."
That's backwards -- visible BENT knees are actually a sitting signature.

Version 3 (this one) looks at leg *posture*, using the pose model's hip,
knee, shoulder, and ankle keypoints:
  - Standing: the knee sits far below the hip (roughly a full thigh-length
    drop) and the leg is close to straight.
  - Sitting (in a car seat, chair, etc.): the knee sits at roughly the
    same height as the hip, because the thigh is close to horizontal.
  - Fully occluded legs (hip/knee not detected at all): can't measure
    posture directly, but a person whose lower body is completely hidden
    while overlapping a car is almost always inside it.

Decision rule, per person:
  1. Find the vehicle (car/bus/truck) they overlap most.
  2. If they don't meaningfully overlap any vehicle -> OUTSIDE (unrelated
     to any car in the photo).
  3. If they do overlap a vehicle significantly, classify by leg posture:
       - Knee roughly level with hip (bent leg)      -> INSIDE
       - Knee well below hip (straight, standing leg) -> OUTSIDE
       - Hip/knee not detected at all (fully hidden)  -> INSIDE
       - Ambiguous / can't tell                       -> OUTSIDE (default)
"""

from dataclasses import dataclass
from typing import List, Literal, Optional
import cv2
from ultralytics import YOLO

# COCO detection class ids
VEHICLE_CLASSES = {2: "car", 5: "bus", 7: "truck"}

# COCO pose keypoint indices
LEFT_SHOULDER, RIGHT_SHOULDER = 5, 6
LEFT_HIP, RIGHT_HIP = 11, 12
LEFT_KNEE, RIGHT_KNEE = 13, 14
LEFT_ANKLE, RIGHT_ANKLE = 15, 16

KPT_CONF_THRESH = 0.5
# knee-drop-below-hip, as a fraction of torso length, below which we call
# the leg "bent" (sitting) rather than "extended" (standing)
SITTING_RATIO_THRESH = 0.55

_det_model = None
_pose_model = None


def get_detector_model():
    """Object detector, used to find cars/buses/trucks."""
    global _det_model
    if _det_model is None:
        _det_model = YOLO("yolov8n.pt")
    return _det_model


def get_pose_model():
    """Pose model, used to find people + their body keypoints."""
    global _pose_model
    if _pose_model is None:
        _pose_model = YOLO("yolov8n-pose.pt")
    return _pose_model


@dataclass
class Box:
    x1: float
    y1: float
    x2: float
    y2: float
    conf: float

    @property
    def area(self):
        return max(0, self.x2 - self.x1) * max(0, self.y2 - self.y1)


@dataclass
class PersonResult:
    box: Box
    status: Literal["inside", "outside"]
    matched_vehicle: Optional[Box]
    overlap_ratio: float
    posture: str  # "sitting" | "standing" | "legs_hidden" | "unrelated" | "unclear"


def _intersection_area(a: Box, b: Box) -> float:
    ix1 = max(a.x1, b.x1)
    iy1 = max(a.y1, b.y1)
    ix2 = min(a.x2, b.x2)
    iy2 = min(a.y2, b.y2)
    return max(0, ix2 - ix1) * max(0, iy2 - iy1)


def _best_overlapping_vehicle(person_box: Box, vehicles: List[Box]):
    best_ratio = 0.0
    best_vehicle = None
    for v in vehicles:
        if person_box.area == 0:
            continue
        inter = _intersection_area(person_box, v)
        ratio = inter / person_box.area
        if ratio > best_ratio:
            best_ratio = ratio
            best_vehicle = v
    return best_vehicle, best_ratio


def _avg_y_if_confident(kpt_xy, kpt_conf, idx_a, idx_b, thresh=KPT_CONF_THRESH):
    """Average the y-coordinate of a left/right keypoint pair, using only
    the side(s) that were detected confidently. Returns None if neither
    side is confident enough to use."""
    ys = []
    for idx in (idx_a, idx_b):
        if float(kpt_conf[idx]) >= thresh:
            ys.append(float(kpt_xy[idx][1]))
    return sum(ys) / len(ys) if ys else None


def _leg_posture(kpt_xy, kpt_conf) -> str:
    """Classify a person's leg posture as 'sitting', 'standing',
    'legs_hidden', or 'unclear', using hip/knee/shoulder keypoints."""
    shoulder_y = _avg_y_if_confident(kpt_xy, kpt_conf, LEFT_SHOULDER, RIGHT_SHOULDER)
    hip_y = _avg_y_if_confident(kpt_xy, kpt_conf, LEFT_HIP, RIGHT_HIP)
    knee_y = _avg_y_if_confident(kpt_xy, kpt_conf, LEFT_KNEE, RIGHT_KNEE)
    ankle_y = _avg_y_if_confident(kpt_xy, kpt_conf, LEFT_ANKLE, RIGHT_ANKLE)

    if hip_y is None and knee_y is None and ankle_y is None:
        # Whole lower body missing from the detection -> hidden, not just
        # posture we can't read.
        return "legs_hidden"

    if hip_y is not None and knee_y is not None and shoulder_y is not None:
        torso_len = abs(hip_y - shoulder_y)
        if torso_len > 1e-3:
            thigh_drop = knee_y - hip_y  # positive: knee below hip, as expected
            sitting_ratio = thigh_drop / torso_len
            if sitting_ratio < SITTING_RATIO_THRESH:
                return "sitting"
            else:
                return "standing"

    if hip_y is not None and knee_y is None and ankle_y is None:
        # Hip visible but nothing below it detected -> legs hidden below
        # the hip (e.g. behind a car door/dashboard).
        return "legs_hidden"

    return "unclear"


def analyze_image(image_path: str, conf_threshold: float = 0.35,
                   overlap_gate: float = 0.35):
    """
    Run detection + classification on an image file.

    overlap_gate: minimum box-overlap ratio for a person to be considered
    "near/on" a vehicle at all. Below this, they're just OUTSIDE (not
    associated with any car in the photo).

    Returns dict with 'people', 'vehicles', 'annotated_image'.
    """
    det_model = get_detector_model()
    pose_model = get_pose_model()

    img_for_size = cv2.imread(image_path)
    if img_for_size is None:
        raise ValueError(f"Could not read image: {image_path}")
    img_h, img_w = img_for_size.shape[:2]
    edge_margin = 6  # pixels

    det_results = det_model.predict(image_path, conf=conf_threshold, verbose=False)[0]
    vehicles = []
    for b in det_results.boxes:
        cls_id = int(b.cls[0])
        if cls_id in VEHICLE_CLASSES:
            x1, y1, x2, y2 = [float(v) for v in b.xyxy[0]]
            vehicles.append(Box(x1, y1, x2, y2, float(b.conf[0])))

    pose_results = pose_model.predict(image_path, conf=conf_threshold, verbose=False)[0]

    people_results: List[PersonResult] = []
    if pose_results.keypoints is not None and len(pose_results.boxes) > 0:
        for i, b in enumerate(pose_results.boxes):
            x1, y1, x2, y2 = [float(v) for v in b.xyxy[0]]
            person_box = Box(x1, y1, x2, y2, float(b.conf[0]))

            vehicle, ratio = _best_overlapping_vehicle(person_box, vehicles)

            if vehicle is None or ratio < overlap_gate:
                people_results.append(PersonResult(
                    box=person_box, status="outside",
                    matched_vehicle=vehicle, overlap_ratio=round(ratio, 3),
                    posture="unrelated",
                ))
                continue

            kpt_xy = pose_results.keypoints.xy[i]
            kpt_conf = pose_results.keypoints.conf[i]
            posture = _leg_posture(kpt_xy, kpt_conf)

            # If the person's box is cropped by the edge of the photo
            # (left/right/bottom), a "legs_hidden" reading could just mean
            # their legs are out of frame, not hidden by the car. Don't
            # trust that signal in that case -- default to outside.
            touches_edge = (
                person_box.x1 <= edge_margin
                or person_box.x2 >= img_w - edge_margin
                or person_box.y2 >= img_h - edge_margin
            )
            if touches_edge and posture == "legs_hidden":
                posture = "unclear"

            status = "inside" if posture in ("sitting", "legs_hidden") else "outside"

            people_results.append(PersonResult(
                box=person_box, status=status,
                matched_vehicle=vehicle, overlap_ratio=round(ratio, 3),
                posture=posture,
            ))

    annotated = _draw_annotations(image_path, people_results, vehicles)

    return {
        "people": people_results,
        "vehicles": vehicles,
        "annotated_image": annotated,
    }


def _draw_annotations(image_path, people_results: List[PersonResult],
                       vehicles: List[Box]):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")

    img_h, img_w = img.shape[:2]
    # scale font/line thickness relative to image size so text stays
    # legible on both small and large photos
    scale = max(img_w, img_h) / 700.0
    box_thickness = max(2, round(3 * scale))
    font_scale = max(0.7, 0.9 * scale)
    text_thickness = max(2, round(2.5 * scale))

    for v in vehicles:
        cv2.rectangle(img, (int(v.x1), int(v.y1)), (int(v.x2), int(v.y2)),
                      (255, 140, 0), box_thickness)

    for pr in people_results:
        color = (0, 200, 0) if pr.status == "inside" else (0, 60, 255)
        b = pr.box
        cv2.rectangle(img, (int(b.x1), int(b.y1)), (int(b.x2), int(b.y2)),
                      color, box_thickness)

        label = f"{pr.status.upper()} CAR"
        (text_w, text_h), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_thickness)

        # place label above the box, or below it if there's no room above
        label_y_top = int(b.y1) - text_h - 14
        if label_y_top < 0:
            label_y_top = int(b.y1) + 6
        label_x = max(0, int(b.x1))

        # solid background behind the text so it's readable over any photo
        cv2.rectangle(
            img,
            (label_x, label_y_top),
            (label_x + text_w + 16, label_y_top + text_h + 14),
            color, -1,
        )
        cv2.putText(
            img, label,
            (label_x + 8, label_y_top + text_h + 4),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255),
            text_thickness, cv2.LINE_AA,
        )

    return img


def summarize(people_results: List[PersonResult]) -> str:
    if not people_results:
        return "No people detected in the image."
    if len(people_results) == 1:
        return f"{people_results[0].status.capitalize()} the car"
    lines = []
    for i, pr in enumerate(people_results, 1):
        lines.append(f"Person {i}: {pr.status.capitalize()} the car")
    return "\n".join(lines)