"""
inspect_images.py
----------------
Run quality control inference on one image or a folder.

Usage:
    python inspect_images.py --image data/raw/PASS/product_001.jpg
    python inspect_images.py --folder data/raw/test_batch/
    python inspect_images.py --folder data/raw/test_batch/ --no_report
"""

import argparse
import yaml
import json

from src.quality_control.inspector import QualityInspector
from src.reporting.report_generator import ReportGenerator


def parse_args():
    p = argparse.ArgumentParser(description="Quality Control Inspector")
    p.add_argument("--config",      default="config.yaml")
    p.add_argument("--model",       default="models/best_model.pth")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--image",   help="Path to single product image")
    group.add_argument("--folder",  help="Path to folder of product images")
    p.add_argument("--no_report",   action="store_true", help="Skip report generation")
    return p.parse_args()


def print_result(result: dict):
    verdict = result["verdict"]
    icon = "PASS" if verdict == "PASS" else "DEFECT"
    print(f"\n{icon}  {verdict}  (confidence: {result['confidence']:.1%})")
    print(f"   Image : {result['image_path']}")
    if result.get("defect_code"):
        dc = result["defect_code"]
        print(f"   Defect: [{dc['code']}] {dc['description']}")
    if result.get("heatmap_path"):
        print(f"   Heatmap -> {result['heatmap_path']}")


def main():
    args = parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    inspector = QualityInspector(args.model, config)

    if args.image:
        result = inspector.inspect(args.image)
        print_result(result)
        results = [result]
    else:
        results = inspector.inspect_folder(args.folder)
        print(f"\n{'='*50}")
        print(f"  Batch Inspection Summary")
        print(f"{'='*50}")
        passed  = sum(1 for r in results if r.get("verdict") == "PASS")
        defects = sum(1 for r in results if r.get("verdict") == "DEFECT")
        print(f"  Total:   {len(results)}")
        print(f"  Pass:  {passed}")
        print(f"  Defect:{defects}")
        print(f"  Pass Rate: {passed/len(results)*100:.1f}%\n")
        for r in results:
            print_result(r)

    if not args.no_report and results:
        reporter = ReportGenerator(config)
        paths = reporter.generate(results)
        print(f"\nReport saved:")
        for fmt, path in paths.items():
            print(f"   [{fmt.upper()}] {path}")


if __name__ == "__main__":
    main()
