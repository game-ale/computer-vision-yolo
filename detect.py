import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Run YOLO detection on webcam/images/videos.")
	parser.add_argument("--weights", default="best.pt", help="Path to YOLO model weights (.pt)")
	parser.add_argument(
		"--source",
		default="0",
		help="Input source: webcam index (e.g. 0), image path, video path, or directory",
	)
	parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold")
	parser.add_argument("--imgsz", type=int, default=640, help="Inference image size")
	parser.add_argument("--show", action="store_true", help="Show live prediction window")
	parser.add_argument("--save", action="store_true", help="Save predictions to disk")
	parser.add_argument("--project", default="outputs", help="Output root directory")
	parser.add_argument("--name", default="predict", help="Output run name")
	return parser.parse_args()


def normalize_source(raw_source: str):
	if raw_source.isdigit():
		return int(raw_source)
	return raw_source


def main() -> None:
	args = parse_args()

	weights_path = Path(args.weights)
	if not weights_path.exists():
		raise FileNotFoundError(f"Model not found: {weights_path.resolve()}")

	model = YOLO(str(weights_path))
	source = normalize_source(args.source)

	model.predict(
		source=source,
		conf=args.conf,
		imgsz=args.imgsz,
		show=args.show,
		save=args.save,
		project=args.project,
		name=args.name,
	)


if __name__ == "__main__":
	main()