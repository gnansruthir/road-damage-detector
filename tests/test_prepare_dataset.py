import xml.etree.ElementTree as ET

from PIL import Image

import prepare_dataset


def test_convert_voc_to_yolo_filters_and_maps_classes(tmp_path):
    annotation = ET.Element("annotation")
    mapped = ET.SubElement(annotation, "object")
    ET.SubElement(mapped, "name").text = "D40"
    box = ET.SubElement(mapped, "bndbox")
    for name, value in [("xmin", "10"), ("ymin", "20"), ("xmax", "50"), ("ymax", "60")]:
        ET.SubElement(box, name).text = value
    ignored = ET.SubElement(annotation, "object")
    ET.SubElement(ignored, "name").text = "D20"

    xml_path = tmp_path / "sample.xml"
    ET.ElementTree(annotation).write(xml_path)

    assert prepare_dataset.convert_voc_to_yolo(xml_path, 100, 100) == [
        "0 0.300000 0.400000 0.400000 0.400000"
    ]


def test_prepare_splits_dry_run_does_not_create_outputs(tmp_path, monkeypatch, capsys):
    source_images = tmp_path / "India" / "train" / "images"
    source_xmls = tmp_path / "India" / "train" / "annotations" / "xmls"
    source_images.mkdir(parents=True)
    source_xmls.mkdir(parents=True)

    image_path = source_images / "sample.jpg"
    Image.new("RGB", (20, 20), (10, 10, 10)).save(image_path)
    annotation = ET.Element("annotation")
    obj = ET.SubElement(annotation, "object")
    ET.SubElement(obj, "name").text = "D00"
    box = ET.SubElement(obj, "bndbox")
    for name, value in [("xmin", "1"), ("ymin", "1"), ("xmax", "10"), ("ymax", "10")]:
        ET.SubElement(box, name).text = value
    ET.ElementTree(annotation).write(source_xmls / "sample.xml")

    monkeypatch.setattr(prepare_dataset, "DATA_DIR", tmp_path)
    prepare_dataset.prepare_splits(limit=1, dry_run=True)

    assert not (tmp_path / "images").exists()
    assert "Dry run: would create 1 night validation images." in capsys.readouterr().out
