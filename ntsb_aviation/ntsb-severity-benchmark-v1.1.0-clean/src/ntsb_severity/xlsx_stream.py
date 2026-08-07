"""Small, dependency-light XLSX reader for tabular NTSB exports.

The project intentionally reads worksheet XML in a streaming manner so the raw
Excel exports can be audited without loading the full workbook into memory.
It supports the cell types used in the supplied files: inline strings, shared
strings, booleans, numbers, and cached formula values.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence
from functools import lru_cache
import re
import zipfile

from lxml import etree

MAIN_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"


@dataclass(frozen=True)
class SheetInfo:
    name: str
    path: str
    dimension: str | None


def column_index(cell_reference: str) -> int:
    """Return a zero-based column index from an A1 cell reference."""
    match = re.match(r"([A-Z]+)", cell_reference)
    if not match:
        raise ValueError(f"Invalid cell reference: {cell_reference!r}")
    value = 0
    for char in match.group(1):
        value = value * 26 + ord(char) - 64
    return value - 1


@lru_cache(maxsize=32)
def list_sheets(path: str | Path) -> list[SheetInfo]:
    path = Path(path)
    with zipfile.ZipFile(path) as archive:
        workbook = etree.fromstring(archive.read("xl/workbook.xml"))
        relationships = etree.fromstring(
            archive.read("xl/_rels/workbook.xml.rels")
        )
        rel_map = {
            node.get("Id"): node.get("Target")
            for node in relationships
        }
        output: list[SheetInfo] = []
        namespaces = {"m": MAIN_NS}
        for sheet in workbook.xpath("//m:sheet", namespaces=namespaces):
            relationship_id = sheet.get(f"{{{REL_NS}}}id")
            target = rel_map[relationship_id]
            if not target.startswith("xl/"):
                target = "xl/" + target.lstrip("/")
            dimension = None
            with archive.open(target) as worksheet_handle:
                for _, node in etree.iterparse(
                    worksheet_handle, events=("end",), tag=f"{{{MAIN_NS}}}dimension"
                ):
                    dimension = node.get("ref")
                    node.clear()
                    break
            output.append(SheetInfo(sheet.get("name"), target, dimension))
        return output


def _load_shared_strings(archive: zipfile.ZipFile) -> list[str]:
    member = "xl/sharedStrings.xml"
    if member not in archive.namelist():
        return []
    output: list[str] = []
    with archive.open(member) as handle:
        for _, element in etree.iterparse(
            handle, events=("end",), tag=f"{{{MAIN_NS}}}si"
        ):
            text = "".join(
                element.xpath(".//m:t/text()", namespaces={"m": MAIN_NS})
            )
            output.append(text)
            element.clear()
    return output


def _sheet_path(path: str | Path, sheet_name: str) -> str:
    for sheet in list_sheets(path):
        if sheet.name == sheet_name:
            return sheet.path
    available = ", ".join(sheet.name for sheet in list_sheets(path))
    raise KeyError(f"Sheet {sheet_name!r} was not found. Available: {available}")


def _cell_value(cell: etree._Element, shared_strings: Sequence[str]):
    cell_type = cell.get("t")
    if cell_type == "inlineStr":
        parts = []
        for node in cell.iter():
            if node.tag == f"{{{MAIN_NS}}}t" and node.text:
                parts.append(node.text)
        return "".join(parts)
    value_node = cell.find(f"{{{MAIN_NS}}}v")
    if value_node is None:
        return None
    text = value_node.text
    if cell_type == "s" and text is not None:
        return shared_strings[int(text)]
    if cell_type == "b":
        return text == "1"
    return text


def iter_rows(
    path: str | Path,
    sheet_name: str,
    *,
    trim_to_header: bool = True,
) -> Iterator[list[object | None]]:
    """Yield worksheet rows as lists.

    Empty cells are returned as ``None``. When ``trim_to_header`` is true, rows
    after the first row are padded or truncated to the header width.
    """
    path = Path(path)
    worksheet_path = _sheet_path(path, sheet_name)
    with zipfile.ZipFile(path) as archive:
        shared_strings = _load_shared_strings(archive)
        header_width: int | None = None
        with archive.open(worksheet_path) as handle:
            for _, row_element in etree.iterparse(
                handle, events=("end",), tag=f"{{{MAIN_NS}}}row"
            ):
                values: dict[int, object | None] = {}
                for cell in row_element.findall(f"{{{MAIN_NS}}}c"):
                    reference = cell.get("r")
                    if reference is None:
                        continue
                    values[column_index(reference)] = _cell_value(
                        cell, shared_strings
                    )
                if values:
                    width = max(values) + 1
                    row: list[object | None] = [None] * width
                    for index, value in values.items():
                        row[index] = value
                else:
                    row = []
                if header_width is None:
                    header_width = len(row)
                elif trim_to_header:
                    if len(row) < header_width:
                        row.extend([None] * (header_width - len(row)))
                    elif len(row) > header_width:
                        row = row[:header_width]
                yield row
                row_element.clear()
                parent = row_element.getparent()
                if parent is not None:
                    while row_element.getprevious() is not None:
                        del parent[0]


def iter_dicts(path: str | Path, sheet_name: str) -> Iterator[dict[str, object | None]]:
    rows = iter_rows(path, sheet_name)
    try:
        header_row = next(rows)
    except StopIteration as exc:
        raise ValueError(f"Worksheet {sheet_name!r} is empty") from exc
    headers = [str(value) if value is not None else "" for value in header_row]
    if len(headers) != len(set(headers)):
        raise ValueError(f"Duplicate headers detected in {sheet_name!r}")
    for row in rows:
        yield dict(zip(headers, row))


def read_header(path: str | Path, sheet_name: str) -> list[str]:
    row = next(iter_rows(path, sheet_name))
    return [str(value) if value is not None else "" for value in row]
