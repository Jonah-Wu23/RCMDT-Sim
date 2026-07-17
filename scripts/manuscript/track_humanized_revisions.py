from __future__ import annotations

import argparse
import datetime as dt
import importlib.util
import zipfile
from copy import deepcopy
from pathlib import Path

from lxml import etree


W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
XML_NS = "http://www.w3.org/XML/1998/namespace"
NS = {"w": W_NS}


def w(tag: str) -> str:
    return f"{{{W_NS}}}{tag}"


def load_replacements(script_path: Path) -> dict[str, str]:
    spec = importlib.util.spec_from_file_location("humanize_user_latest_manuscript", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return dict(module.REPLACEMENTS)


def visible_text(paragraph: etree._Element) -> str:
    return "".join(paragraph.xpath(".//w:t/text()", namespaces=NS))


def normalize(text: str) -> str:
    return " ".join(text.split())


def first_run_properties(paragraph: etree._Element) -> etree._Element | None:
    node = paragraph.find(".//w:r/w:rPr", namespaces=NS)
    return deepcopy(node) if node is not None else None


def make_run(text: str, rpr: etree._Element | None, deleted: bool) -> etree._Element:
    run = etree.Element(w("r"))
    if rpr is not None:
        run.append(deepcopy(rpr))
    node = etree.SubElement(run, w("delText" if deleted else "t"))
    node.set(f"{{{XML_NS}}}space", "preserve")
    node.text = text
    return run


def next_revision_id(root: etree._Element) -> int:
    ids: list[int] = []
    for node in root.xpath(".//*[@w:id]", namespaces=NS):
        try:
            ids.append(int(node.get(w("id"))))
        except (TypeError, ValueError):
            pass
    return max(ids, default=0) + 1


def track_replacements(
    source: Path,
    output: Path,
    replacements: dict[str, str],
    author: str,
) -> None:
    with zipfile.ZipFile(source, "r") as archive:
        document_root = etree.fromstring(archive.read("word/document.xml"))
        settings_root = etree.fromstring(archive.read("word/settings.xml"))
        if settings_root.find("w:trackRevisions", namespaces=NS) is None:
            settings_root.insert(0, etree.Element(w("trackRevisions")))

        revision_id = next_revision_id(document_root)
        when = dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
        matched: set[str] = set()

        for paragraph in document_root.xpath(".//w:body/w:p", namespaces=NS):
            old_text = visible_text(paragraph)
            normalized = normalize(old_text)
            prefix = next((key for key in replacements if normalized.startswith(key)), None)
            if prefix is None:
                continue

            rpr = first_run_properties(paragraph)
            ppr = paragraph.find("w:pPr", namespaces=NS)
            for child in list(paragraph):
                if child is not ppr:
                    paragraph.remove(child)

            deletion = etree.SubElement(paragraph, w("del"))
            deletion.set(w("id"), str(revision_id))
            deletion.set(w("author"), author)
            deletion.set(w("date"), when)
            deletion.append(make_run(old_text, rpr, deleted=True))
            revision_id += 1

            insertion = etree.SubElement(paragraph, w("ins"))
            insertion.set(w("id"), str(revision_id))
            insertion.set(w("author"), author)
            insertion.set(w("date"), when)
            insertion.append(make_run(replacements[prefix], rpr, deleted=False))
            revision_id += 1
            matched.add(prefix)

        missing = set(replacements) - matched
        if missing:
            raise RuntimeError(f"Could not find paragraphs for {sorted(missing)}")

        overrides = {
            "word/document.xml": etree.tostring(
                document_root, xml_declaration=True, encoding="UTF-8", standalone="yes"
            ),
            "word/settings.xml": etree.tostring(
                settings_root, xml_declaration=True, encoding="UTF-8", standalone="yes"
            ),
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as target:
            for info in archive.infolist():
                target.writestr(info, overrides.get(info.filename, archive.read(info.filename)))

    print(f"{output} replacements={len(matched)} revisions={len(matched) * 2}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replacement-script", type=Path, required=True)
    parser.add_argument("--author", default="Codex")
    args = parser.parse_args()
    replacements = load_replacements(args.replacement_script.resolve())
    track_replacements(args.source.resolve(), args.output.resolve(), replacements, args.author)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
