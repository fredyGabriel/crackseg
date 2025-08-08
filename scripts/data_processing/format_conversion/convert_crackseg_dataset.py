#!/usr/bin/env python3
"""
Batch conversion script for CrackSeg dataset.

This script converts the BD_estudio segmentation dataset to object detection format
with proper organization and validation.

Author: CrackSeg Project
Date: 2025-01-13
"""

import argparse
import sys
from pathlib import Path
from typing import Any

# Add the script directory to Python path for imports
sys.path.append(str(Path(__file__).parent))

from segmentation_to_detection import MaskToDetectionConverter


def organize_detection_dataset(
    source_image_dir: Path,
    source_mask_dir: Path,
    output_base_dir: Path,
    formats: list[str] | None = None,
) -> dict[str, Path]:
    """
    Organize detection dataset with proper structure.

    Args:
        source_image_dir: Directory with original images
        source_mask_dir: Directory with segmentation masks
        output_base_dir: Base output directory
        formats: List of output formats to create ('yolo', 'coco', 'pascal_voc')

    Returns:
        Dictionary with organized directory paths
    """
    if formats is None:
        formats = ["yolo"]

    # Create organized structure (simplified)
    organized_dirs = {
        "base": output_base_dir,
        "images": source_image_dir,  # Reference to original images
        "annotations": output_base_dir,  # Direct output directory
    }

    # Create only necessary directories
    output_base_dir.mkdir(parents=True, exist_ok=True)

    print(f"Organizing dataset structure in: {output_base_dir}")
    print(f"Using original images from: {source_image_dir}")
    print("No image copying - using direct references to save space")

    return organized_dirs


def create_dataset_info_file(output_dir: Path, stats: dict[str, Any]) -> None:
    """Create a dataset information file."""
    info_file = output_dir / "dataset_info.txt"

    with open(info_file, "w") as f:
        f.write("CrackSeg Object Detection Dataset\n")
        f.write("=" * 40 + "\n\n")
        f.write("Dataset Statistics:\n")
        f.write(f"- Total images: {stats.get('total_images', 0)}\n")
        f.write(
            f"- Images with detections: {stats.get('images_with_detections', 0)}\n"
        )
        f.write(f"- Total detections: {stats.get('total_detections', 0)}\n")
        f.write(
            f"- Failed conversions: {stats.get('failed_conversions', 0)}\n"
        )

        if stats.get("total_images", 0) > 0:
            success_rate = (
                stats.get("images_with_detections", 0) / stats["total_images"]
            ) * 100
            f.write(f"- Success rate: {success_rate:.1f}%\n")

        if stats.get("images_with_detections", 0) > 0:
            avg_detections = (
                stats["total_detections"] / stats["images_with_detections"]
            )
            f.write(f"- Average detections per image: {avg_detections:.1f}\n")

        f.write("\nClass Information:\n")
        f.write("- Class ID: 0\n")
        f.write("- Class Name: crack\n")
        f.write("- Type: Pavement crack segmentation\n")

        f.write("\nFile Structure:\n")
        f.write("- images/ : Original images (referenced)\n")
        f.write("- yolo/ : YOLO format annotations\n")
        f.write("- classes.txt : Class names (YOLO format)\n")
        f.write("- dataset_info.txt : This file\n")

    print(f"Dataset info saved to: {info_file}")


def validate_conversion(
    image_dir: Path, annotation_dir: Path, format_type: str
) -> dict[str, int]:
    """
    Validate the conversion results.

    Args:
        image_dir: Directory with images
        annotation_dir: Directory with annotations
        format_type: Annotation format

    Returns:
        Validation statistics
    """
    print("Validating conversion results...")

    # Get all image files
    image_files = []
    for ext in [".jpg", ".jpeg", ".png", ".bmp"]:
        image_files.extend(list(image_dir.glob(f"*{ext}")))

    # Get all annotation files
    if format_type == "yolo":
        annotation_files = list(annotation_dir.glob("*.txt"))
        # Exclude classes.txt
        annotation_files = [
            f for f in annotation_files if f.name != "classes.txt"
        ]
    elif format_type == "pascal_voc":
        annotation_files = list(annotation_dir.glob("*.xml"))
    elif format_type == "coco":
        annotation_files = list(annotation_dir.glob("*.json"))
    else:
        annotation_files = []

    # Check for matching pairs
    image_stems = {f.stem for f in image_files}
    annotation_stems = {f.stem for f in annotation_files}

    matched = image_stems & annotation_stems
    missing_annotations = image_stems - annotation_stems
    orphaned_annotations = annotation_stems - image_stems

    validation_stats = {
        "total_images": len(image_files),
        "total_annotations": len(annotation_files),
        "matched_pairs": len(matched),
        "missing_annotations": len(missing_annotations),
        "orphaned_annotations": len(orphaned_annotations),
    }

    print("Validation Results:")
    print(f"- Total images: {validation_stats['total_images']}")
    print(f"- Total annotations: {validation_stats['total_annotations']}")
    print(f"- Matched pairs: {validation_stats['matched_pairs']}")
    print(f"- Missing annotations: {validation_stats['missing_annotations']}")
    print(
        f"- Orphaned annotations: {validation_stats['orphaned_annotations']}"
    )

    if missing_annotations:
        print(
            f"- Images without annotations: {list(missing_annotations)[:5]}..."
        )

    return validation_stats


def main() -> None:
    """Main function for batch conversion."""
    parser = argparse.ArgumentParser(
        description="Convert CrackSeg segmentation dataset to object detection format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert to YOLO format (default)
  python convert_crackseg_dataset.py \\
    --output-dir data/BD_estudio/2-Object_detection

  # Convert to COCO format
  python convert_crackseg_dataset.py \\
    --format coco \\
    --output-dir data/BD_estudio/2-Object_detection_coco

  # Custom source directories
  python convert_crackseg_dataset.py \\
    --image-dir path/to/images \\
    --mask-dir path/to/masks \\
    --output-dir path/to/output \\
    --format yolo
        """,
    )

    parser.add_argument(
        "--image-dir",
        type=str,
        default="data/BD_estudio/1-Segmentation/Original image",
        help="Directory containing original images",
    )

    parser.add_argument(
        "--mask-dir",
        type=str,
        default="data/BD_estudio/1-Segmentation/Ground truth",
        help="Directory containing segmentation masks",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save detection dataset",
    )

    parser.add_argument(
        "--format",
        type=str,
        choices=["yolo", "coco", "pascal_voc"],
        default="yolo",
        help="Output annotation format (default: yolo)",
    )

    parser.add_argument(
        "--class-name",
        type=str,
        default="crack",
        help="Name of the object class (default: crack)",
    )

    parser.add_argument(
        "--organize",
        action="store_true",
        help="Organize dataset with standard structure",
    )

    parser.add_argument(
        "--validate", action="store_true", help="Validate conversion results"
    )

    args = parser.parse_args()

    # Convert paths
    image_dir = Path(args.image_dir)
    mask_dir = Path(args.mask_dir)
    output_dir = Path(args.output_dir)

    print("CrackSeg Dataset Conversion")
    print("=" * 50)
    print(f"Source images: {image_dir}")
    print(f"Source masks: {mask_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Format: {args.format.upper()}")
    print(f"Class name: {args.class_name}")

    try:
        # Organize dataset structure if requested
        if args.organize:
            organized_dirs = organize_detection_dataset(
                image_dir, mask_dir, output_dir, args.format
            )
            annotation_output_dir = organized_dirs["annotations"]
        else:
            output_dir.mkdir(parents=True, exist_ok=True)
            annotation_output_dir = output_dir

        # Create converter and run conversion
        converter = MaskToDetectionConverter(
            image_dir=image_dir,
            mask_dir=mask_dir,
            output_dir=annotation_output_dir,
            format_type=args.format,
            class_name=args.class_name,
            class_id=0,
        )

        converter.convert_dataset()

        # Create dataset info file
        create_dataset_info_file(output_dir, converter.stats)

        # Validate conversion if requested
        if args.validate:
            if args.organize:
                validate_conversion(
                    organized_dirs["images"],
                    annotation_output_dir,
                    args.format,
                )
            else:
                validate_conversion(
                    image_dir, annotation_output_dir, args.format
                )

        print("\n✅ Conversion completed successfully!")
        print(f"📁 Output saved to: {output_dir}")

        if args.format == "yolo":
            print("\n📋 YOLO Training Usage:")
            print("   Use the following structure for training:")
            if args.organize:
                print(f"   - Images: {organized_dirs['images']}")
                print(f"   - Labels: {annotation_output_dir}")
            else:
                print(f"   - Images: {image_dir}")
                print(f"   - Labels: {annotation_output_dir}")

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
