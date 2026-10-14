"""
A deliberately small PDF writer — just enough for a 1-2 page evidence
report: Helvetica / Helvetica-Bold text (the PDF standard fonts, so nothing
is embedded), filled/stroked rectangles and lines, and JPEG images.

Why not a PDF library? None is in requirements.txt, and the usual choices
pull in Pillow and several other packages. OpenCV already gives us JPEG
bytes, and PDF can embed JPEG as-is (DCTDecode), so ~200 lines of plain
Python keep the project dependency-free and fully offline.

Coordinates in this API are in points from the TOP-left corner of the page
(y grows downward); they are converted to PDF's bottom-left origin inside.
"""
from __future__ import annotations

import zlib

A4 = (595.28, 841.89)

# Glyph widths (1/1000 em) for ASCII 32..126 from the Adobe core-font AFMs.
_HELV = [
    278, 278, 355, 556, 556, 889, 667, 191, 333, 333, 389, 584, 278, 333, 278, 278,
    556, 556, 556, 556, 556, 556, 556, 556, 556, 556, 278, 278, 584, 584, 584, 556,
    1015, 667, 667, 722, 722, 667, 611, 778, 722, 278, 500, 667, 556, 833, 722, 778,
    667, 778, 722, 667, 611, 722, 667, 944, 667, 667, 611, 278, 278, 278, 469, 556,
    333, 556, 556, 500, 556, 556, 278, 556, 556, 222, 222, 500, 222, 833, 556, 556,
    556, 556, 333, 500, 278, 556, 500, 722, 500, 500, 500, 334, 260, 334, 584,
]
_HELV_BOLD = [
    278, 333, 474, 556, 556, 889, 722, 238, 333, 333, 389, 584, 278, 333, 278, 278,
    556, 556, 556, 556, 556, 556, 556, 556, 556, 556, 333, 333, 584, 584, 584, 611,
    975, 722, 722, 722, 722, 667, 611, 778, 722, 278, 556, 722, 611, 833, 722, 778,
    667, 778, 722, 667, 611, 722, 667, 944, 667, 667, 611, 333, 278, 333, 584, 556,
    333, 556, 611, 556, 611, 556, 333, 611, 611, 278, 278, 556, 278, 889, 611, 611,
    611, 611, 389, 556, 333, 611, 556, 778, 556, 556, 500, 389, 280, 389, 584,
]

# Characters outside WinAnsi (cp1252) that we do use elsewhere in the app.
_REPLACE = {"↔": "-", "→": "->", "≤": "<=", "≥": ">=", "−": "-"}


def _to_winansi(s: str) -> bytes:
    for k, v in _REPLACE.items():
        s = s.replace(k, v)
    return s.encode("cp1252", errors="replace")


def _esc(b: bytes) -> bytes:
    return b.replace(b"\\", b"\\\\").replace(b"(", b"\\(").replace(b")", b"\\)")


def hex_rgb(h: str) -> tuple[float, float, float]:
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))  # type: ignore[return-value]


class PdfDocument:
    def __init__(self, size=A4) -> None:
        self.w, self.h = size
        self._pages: list[list[bytes]] = []
        self._images: list[tuple[bytes, int, int]] = []
        self.add_page()

    # ---- pages ---------------------------------------------------------------
    def add_page(self) -> None:
        self._pages.append([])

    @property
    def page_count(self) -> int:
        return len(self._pages)

    def _op(self, s: str | bytes) -> None:
        self._pages[-1].append(s.encode("latin-1") if isinstance(s, str) else s)

    # ---- measurement -----------------------------------------------------------
    @staticmethod
    def text_width(s: str, size: float, bold: bool = False) -> float:
        table = _HELV_BOLD if bold else _HELV
        total = 0
        for ch in _to_winansi(s):
            total += table[ch - 32] if 32 <= ch <= 126 else 556
        return total * size / 1000

    def wrap(self, s: str, size: float, max_w: float, bold: bool = False) -> list[str]:
        lines: list[str] = []
        for para in s.split("\n"):
            cur = ""
            for word in para.split(" "):
                cand = word if not cur else cur + " " + word
                if self.text_width(cand, size, bold) <= max_w or not cur:
                    cur = cand
                else:
                    lines.append(cur)
                    cur = word
            lines.append(cur)
        return lines

    # ---- drawing ------------------------------------------------------------
    def text(self, x: float, y: float, s: str, size: float = 10, bold: bool = False,
             color: str = "#10231b", align: str = "left") -> None:
        """y is the text BASELINE, measured from the top of the page."""
        if align != "left":
            w = self.text_width(s, size, bold)
            x = x - w if align == "right" else x - w / 2
        r, g, b = hex_rgb(color)
        font = "F2" if bold else "F1"
        self._op(f"BT {r:.3f} {g:.3f} {b:.3f} rg /{font} {size:.2f} Tf {x:.2f} {self.h - y:.2f} Td (".encode("latin-1")
                 + _esc(_to_winansi(s)) + b") Tj ET\n")

    def paragraph(self, x: float, y: float, s: str, size: float, max_w: float, bold: bool = False,
                  color: str = "#10231b", leading: float = 1.35) -> float:
        """Draw wrapped text starting at baseline y; returns the y after the last line."""
        for line in self.wrap(s, size, max_w, bold):
            self.text(x, y, line, size, bold, color)
            y += size * leading
        return y

    def rect(self, x: float, y: float, w: float, h: float, fill: str | None = None,
             stroke: str | None = None, lw: float = 0.75) -> None:
        ops = []
        if fill:
            ops.append("%.3f %.3f %.3f rg" % hex_rgb(fill))
        if stroke:
            ops.append("%.3f %.3f %.3f RG %.2f w" % (*hex_rgb(stroke), lw))
        paint = "B" if (fill and stroke) else ("f" if fill else "S")
        ops.append(f"{x:.2f} {self.h - y - h:.2f} {w:.2f} {h:.2f} re {paint}")
        self._op("q " + " ".join(ops) + " Q\n")

    def line(self, x1: float, y1: float, x2: float, y2: float, color: str = "#dfdccf", lw: float = 0.75) -> None:
        r, g, b = hex_rgb(color)
        self._op(f"q {r:.3f} {g:.3f} {b:.3f} RG {lw:.2f} w {x1:.2f} {self.h - y1:.2f} m {x2:.2f} {self.h - y2:.2f} l S Q\n")

    def jpeg(self, data: bytes, px_w: int, px_h: int, x: float, y: float, w: float, h: float) -> None:
        self._images.append((data, px_w, px_h))
        name = f"Im{len(self._images)}"
        self._op(f"q {w:.2f} 0 0 {h:.2f} {x:.2f} {self.h - y - h:.2f} cm /{name} Do Q\n")

    # ---- output -------------------------------------------------------------
    def to_bytes(self) -> bytes:
        objs: list[bytes] = []

        def add(body: bytes) -> int:
            objs.append(body)
            return len(objs)

        catalog = add(b"")  # placeholder, filled below
        pages = add(b"")
        f1 = add(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>")
        f2 = add(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold /Encoding /WinAnsiEncoding >>")
        img_ids = []
        for data, pw, ph in self._images:
            img_ids.append(add(
                f"<< /Type /XObject /Subtype /Image /Width {pw} /Height {ph} /ColorSpace /DeviceRGB "
                f"/BitsPerComponent 8 /Filter /DCTDecode /Length {len(data)} >>\nstream\n".encode("latin-1")
                + data + b"\nendstream"
            ))
        xobjects = " ".join(f"/Im{i + 1} {oid} 0 R" for i, oid in enumerate(img_ids))
        resources = f"<< /Font << /F1 {f1} 0 R /F2 {f2} 0 R >> /XObject << {xobjects} >> >>"

        page_ids = []
        for ops in self._pages:
            raw = b"".join(ops)
            comp = zlib.compress(raw)
            content = add(f"<< /Length {len(comp)} /Filter /FlateDecode >>\nstream\n".encode("latin-1") + comp + b"\nendstream")
            page_ids.append(add(
                f"<< /Type /Page /Parent {pages} 0 R /MediaBox [0 0 {self.w:.2f} {self.h:.2f}] "
                f"/Resources {resources} /Contents {content} 0 R >>".encode("latin-1")
            ))
        objs[catalog - 1] = f"<< /Type /Catalog /Pages {pages} 0 R >>".encode("latin-1")
        kids = " ".join(f"{p} 0 R" for p in page_ids)
        objs[pages - 1] = f"<< /Type /Pages /Kids [{kids}] /Count {len(page_ids)} >>".encode("latin-1")

        out = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
        offsets = []
        for i, body in enumerate(objs, start=1):
            offsets.append(len(out))
            out += f"{i} 0 obj\n".encode("latin-1") + body + b"\nendobj\n"
        xref = len(out)
        out += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode("latin-1")
        for off in offsets:
            out += f"{off:010d} 00000 n \n".encode("latin-1")
        out += f"trailer\n<< /Size {len(objs) + 1} /Root {catalog} 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode("latin-1")
        return bytes(out)
