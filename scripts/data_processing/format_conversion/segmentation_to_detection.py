#!/usr/bin/env python3
"""
Script to convert segmentation masks to object detection annotations.

This script processes binary segmentation masks and generates bounding box annotations
in various formats (YOLO, COCO, Pascal VOC) for crack detection training.

Author: CrackSeg Project
Date: 2025-01-13
"""

import argparse
import json
import warnings
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from tqdm import tqdm


class MaskToDetectionConverter:
    """Convert segmentation masks to object detection annotations."""

    def __init__(
        self,
        image_dir: Path,
        mask_dir: Path,
        output_dir: Path,
        format_type: str = "yolo",
        class_name: str = "crack",
        class_id: int = 0,
    ) -> None:
        """
        Initialize the converter.

        Args:
            image_dir: Directory containing original images
            mask_dir: Directory containing segmentation masks
            output_dir: Directory to save detection annotations
            format_type: Output format ('yolo', 'coco', 'pascal_voc')
            class_name: Name of the object class
            class_id: ID of the object class (for YOLO/COCO)
        """
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.base_output_dir = Path(output_dir)
        self.format_type = format_type.lower()
        self.class_name = class_name
        self.class_id = class_id

        # Validate inputs
        self._validate_inputs()

        # Create format-specific output directory
        self.output_dir = self.base_output_dir / self.format_type
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize format-specific variables
        if self.format_type == "coco":
            self.coco_data = self._initialize_coco_format()

        # Statistics tracking
        self.stats = {
            "total_images": 0,
            "images_with_detections": 0,
            "total_detections": 0,
            "failed_conversions": 0,
        }

    def _validate_inputs(self) -> None:
        """Validate input parameters and directories."""
        if not self.image_dir.exists():
            raise ValueError(
                f"Image directory does not exist: {self.image_dir}"
            )

        if not self.mask_dir.exists():
            raise ValueError(f"Mask directory does not exist: {self.mask_dir}")

        if self.format_type not in ["yolo", "coco", "pascal_voc"]:
            raise ValueError(f"Unsupported format: {self.format_type}")

        if self.class_id < 0:
            raise ValueError(f"Class ID must be non-negative: {self.class_id}")

    def _initialize_coco_format(self) -> dict[str, Any]:
        """Initialize COCO format structure."""
        return {
            "info": {
                "description": "CrackSeg Object Detection Dataset",
                "version": "1.0",
                "year": 2025,
                "contributor": "CrackSeg Project",
                "date_created": "2025-01-13",
            },
            "licenses": [{"id": 1, "name": "Research License", "url": ""}],
            "images": [],
            "annotations": [],
            "categories": [
                {
                    "id": self.class_id,
                    "name": self.class_name,
                    "supercategory": "defect",
                }
            ],
        }

    def find_contours_from_mask(
        self,
        mask: np.ndarray,
        min_area: int = 50,
        approximation_epsilon: float = 0.002,
    ) -> list[np.ndarray]:
        """
        Extract contours from binary mask.

        Args:
            mask: Binary mask image (0 and 255 values)
            min_area: Minimum contour area to consider
            approximation_epsilon: Epsilon parameter for contour approximation

        Returns:
            List of contour arrays
        """
        # Ensure mask is binary
        if len(mask.shape) == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

        # Threshold to ensure binary values
        _, mask_binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

        # Find contours
        contours, _ = cv2.findContours(
            mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Filter contours by area and approximate them
        valid_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area >= min_area:
                # Approximate contour to reduce points
                epsilon = approximation_epsilon * cv2.arcLength(contour, True)
                approx_contour = cv2.approxPolyDP(contour, epsilon, True)
                valid_contours.append(approx_contour)

        return valid_contours

    def contour_to_bbox(
        self, contour: np.ndarray
    ) -> tuple[int, int, int, int]:
        """
        Convert contour to bounding box coordinates.

        Args:
            contour: Contour array

        Returns:
            Bounding box as (x, y, width, height)
        """
        return cv2.boundingRect(contour)

    def bbox_to_yolo_format(
        self,
        bbox: tuple[int, int, int, int],
        image_width: int,
        image_height: int,
    ) -> tuple[float, float, float, float]:
        """
        Convert bounding box to YOLO format (normalized).

        Args:
            bbox: Bounding box as (x, y, width, height)
            image_width: Image width in pixels
            image_height: Image height in pixels

        Returns:
            YOLO format: (center_x, center_y, width, height) normalized
        """
        x, y, w, h = bbox

        # Calculate center coordinates
        center_x = x + w / 2.0
        center_y = y + h / 2.0

        # Normalize coordinates
        center_x_norm = center_x / image_width
        center_y_norm = center_y / image_height
        width_norm = w / image_width
        height_norm = h / image_height

        return center_x_norm, center_y_norm, width_norm, height_norm

    def process_single_mask(
        self, mask_path: Path, image_path: Path | None = None
    ) -> list[dict[str, Any]]:
        """
        Process a single mask file and extract detections.

        Args:
            mask_path: Path to mask file
            image_path: Optional path to corresponding image

        Returns:
            List of detection dictionaries
        """
        try:
            # Load mask
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                warnings.warn(
                    f"Could not load mask: {mask_path}", stacklevel=2
                )
                return []

            # Get image dimensions
            if image_path and image_path.exists():
                image = cv2.imread(str(image_path))
                if image is not None:
                    height, width = image.shape[:2]
                else:
                    height, width = mask.shape[:2]
            else:
                height, width = mask.shape[:2]

            # Find contours
            contours = self.find_contours_from_mask(mask)

            # Convert contours to detections
            detections = []
            for contour in contours:
                bbox = self.contour_to_bbox(contour)
                _x, _y, w, h = bbox

                # Skip very small detections
                if w < 5 or h < 5:
                    continue

                detection = {
                    "bbox": bbox,
                    "bbox_yolo": self.bbox_to_yolo_format(bbox, width, height),
                    "contour": contour,
                    "area": cv2.contourArea(contour),
                    "image_width": width,
                    "image_height": height,
                }
                detections.append(detection)

            return detections

        except Exception as e:
            warnings.warn(
                f"Error processing mask {mask_path}: {str(e)}", stacklevel=2
            )
            return []

    def save_yolo_annotation(
        self, detections: list[dict[str, Any]], output_path: Path
    ) -> None:
        """Save detections in YOLO format."""
        with open(output_path, "w") as f:
            for detection in detections:
                center_x, center_y, width, height = detection["bbox_yolo"]
                f.write(
                    f"{self.class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}\n"
                )

    def save_coco_annotations(self, output_path: Path) -> None:
        """Save all detections in COCO format."""
        with open(output_path, "w") as f:
            json.dump(self.coco_data, f, indent=2)

    def save_pascal_voc_annotation(
        self,
        detections: list[dict[str, Any]],
        output_path: Path,
        image_filename: str,
    ) -> None:
        """Save detections in Pascal VOC format."""
        if not detections:
            return

        # Get image dimensions from first detection
        height = detections[0]["image_height"]
        width = detections[0]["image_width"]

        # Create XML structure
        annotation = ET.Element("annotation")

        # Folder
        folder = ET.SubElement(annotation, "folder")
        folder.text = "images"

        # Filename
        filename = ET.SubElement(annotation, "filename")
        filename.text = image_filename

        # Size
        size = ET.SubElement(annotation, "size")
        width_elem = ET.SubElement(size, "width")
        width_elem.text = str(width)
        height_elem = ET.SubElement(size, "height")
        height_elem.text = str(height)
        depth_elem = ET.SubElement(size, "depth")
        depth_elem.text = "3"

        # Objects
        for detection in detections:
            obj = ET.SubElement(annotation, "object")

            name = ET.SubElement(obj, "name")
            name.text = self.class_name

            bndbox = ET.SubElement(obj, "bndbox")
            x, y, w, h = detection["bbox"]

            xmin = ET.SubElement(bndbox, "xmin")
            xmin.text = str(x)
            ymin = ET.SubElement(bndbox, "ymin")
            ymin.text = str(y)
            xmax = ET.SubElement(bndbox, "xmax")
            xmax.text = str(x + w)
            ymax = ET.SubElement(bndbox, "ymax")
            ymax.text = str(y + h)

        # Write XML file
        tree = ET.ElementTree(annotation)
        tree.write(output_path, encoding="utf-8", xml_declaration=True)

    def add_to_coco_format(
        self,
        detections: list[dict[str, Any]],
        image_filename: str,
        image_id: int,
    ) -> None:
        """Add detections to COCO format structure."""
        if not detections:
            return

        # Add image info
        height = detections[0]["image_height"]
        width = detections[0]["image_width"]

        image_info = {
            "id": image_id,
            "width": width,
            "height": height,
            "file_name": image_filename,
        }
        self.coco_data["images"].append(image_info)

        # Add annotations
        for detection in detections:
            x, y, w, h = detection["bbox"]
            area = detection["area"]

            annotation = {
                "id": len(self.coco_data["annotations"]),
                "image_id": image_id,
                "category_id": self.class_id,
                "bbox": [x, y, w, h],
                "area": float(area),
                "iscrowd": 0,
            }
            self.coco_data["annotations"].append(annotation)

    def convert_dataset(self) -> None:
        """Convert entire dataset from segmentation to detection format."""
        print(f"Converting dataset to {self.format_type.upper()} format...")
        print(f"Input: {self.mask_dir}")
        print(f"Output: {self.output_dir}")

        # Find all mask files
        mask_files = list(self.mask_dir.glob("*.png")) + list(
            self.mask_dir.glob("*.jpg")
        )

        if not mask_files:
            raise ValueError(f"No mask files found in {self.mask_dir}")

        print(f"Found {len(mask_files)} mask files")

        # Create classes file for YOLO
        if self.format_type == "yolo":
            classes_file = self.output_dir / "classes.txt"
            with open(classes_file, "w") as f:
                f.write(f"{self.class_name}\n")

        # Process each mask
        image_id = 0
        for mask_path in tqdm(mask_files, desc="Processing masks"):
            self.stats["total_images"] += 1

            # Find corresponding image
            image_name = mask_path.stem

            # Try different image extensions
            image_path = None
            for ext in [".jpg", ".jpeg", ".png", ".bmp"]:
                potential_path = self.image_dir / f"{image_name}{ext}"
                if potential_path.exists():
                    image_path = potential_path
                    break

            # Process mask
            detections = self.process_single_mask(mask_path, image_path)

            if detections:
                self.stats["images_with_detections"] += 1
                self.stats["total_detections"] += len(detections)

                # Save in requested format
                if self.format_type == "yolo":
                    output_file = self.output_dir / f"{image_name}.txt"
                    self.save_yolo_annotation(detections, output_file)

                elif self.format_type == "pascal_voc":
                    output_file = self.output_dir / f"{image_name}.xml"
                    image_filename = (
                        f"{image_name}.jpg"  # Assume jpg extension
                    )
                    self.save_pascal_voc_annotation(
                        detections, output_file, image_filename
                    )

                elif self.format_type == "coco":
                    image_filename = (
                        f"{image_name}.jpg"  # Assume jpg extension
                    )
                    self.add_to_coco_format(
                        detections, image_filename, image_id
                    )
                    image_id += 1
            else:
                self.stats["failed_conversions"] += 1

        # Save COCO format if selected
        if self.format_type == "coco":
            coco_output = self.output_dir / "annotations.json"
            self.save_coco_annotations(coco_output)

        # Create dataset info file
        self.create_dataset_info()

        # Print statistics
        self.print_conversion_stats()

    def create_dataset_info(self) -> None:
        """Create dataset information file with conversion details."""
        info_file = self.base_output_dir / "dataset_info.json"

        dataset_info = {
            "dataset_name": "CrackSeg Object Detection Dataset",
            "conversion_date": "2025-01-13",
            "source_segmentation": {
                "image_dir": str(self.image_dir),
                "mask_dir": str(self.mask_dir),
            },
            "formats_available": [self.format_type],
            "class_info": {
                "class_name": self.class_name,
                "class_id": self.class_id,
                "total_classes": 1,
            },
            "statistics": self.stats,
            "directory_structure": {
                self.format_type: {
                    "location": str(self.output_dir),
                    "description": f"Annotations in {self.format_type.upper()} format",
                    "file_pattern": self._get_file_pattern(),
                }
            },
        }

        with open(info_file, "w") as f:
            json.dump(dataset_info, f, indent=2)

        print(f"Dataset info saved to: {info_file}")

    def _get_file_pattern(self) -> str:
        """Get file pattern description for current format."""
        patterns = {
            "yolo": "*.txt files + classes.txt",
            "coco": "annotations.json",
            "pascal_voc": "*.xml files",
        }
        return patterns.get(self.format_type, "unknown")

    def print_conversion_stats(self) -> None:
        """Print conversion statistics."""
        print("\n" + "=" * 50)
        print("CONVERSION STATISTICS")
        print("=" * 50)
        print(f"Total images processed: {self.stats['total_images']}")
        print(
            f"Images with detections: {self.stats['images_with_detections']}"
        )
        print(f"Total detections found: {self.stats['total_detections']}")
        print(f"Failed conversions: {self.stats['failed_conversions']}")

        if self.stats["total_images"] > 0:
            success_rate = (
                self.stats["images_with_detections"]
                / self.stats["total_images"]
            ) * 100
            print(f"Success rate: {success_rate:.1f}%")

        if self.stats["images_with_detections"] > 0:
            avg_detections = (
                self.stats["total_detections"]
                / self.stats["images_with_detections"]
            )
            print(f"Average detections per image: {avg_detections:.1f}")

        print(f"Output directory: {self.output_dir}")


def main() -> None:
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(
        description="Convert segmentation masks to object detection annotations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert to YOLO format
  python segmentation_to_detection.py \\
    --image-dir data/BD_estudio/1-Segmentation/Original\\ image \\
    --mask-dir data/BD_estudio/1-Segmentation/Ground\\ truth \\
    --output-dir data/BD_estudio/2-Object\\ detection \\
    --format yolo

  # Convert to COCO format
  python segmentation_to_detection.py \\
    --image-dir data/BD_estudio/1-Segmentation/Original\\ image \\
    --mask-dir data/BD_estudio/1-Segmentation/Ground\\ truth \\
    --output-dir data/BD_estudio/2-Object\\ detection \\
    --format coco \\
    --class-name crack
        """,
    )

    parser.add_argument(
        "--image-dir",
        type=str,
        required=True,
        help="Directory containing original images",
    )

    parser.add_argument(
        "--mask-dir",
        type=str,
        required=True,
        help="Directory containing segmentation masks",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save detection annotations",
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
        "--class-id",
        type=int,
        default=0,
        help="ID of the object class for YOLO/COCO (default: 0)",
    )

    args = parser.parse_args()

    # Convert paths
    image_dir = Path(args.image_dir)
    mask_dir = Path(args.mask_dir)
    output_dir = Path(args.output_dir)

    try:
        # Create converter and run conversion
        converter = MaskToDetectionConverter(
            image_dir=image_dir,
            mask_dir=mask_dir,
            output_dir=output_dir,
            format_type=args.format,
            class_name=args.class_name,
            class_id=args.class_id,
        )

        converter.convert_dataset()

        print("\nConversion completed successfully!")
        print(f"Output saved to: {output_dir}")

    except Exception as e:
        print(f"Error: {str(e)}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
