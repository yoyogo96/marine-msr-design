#!/usr/bin/env python3
"""
40 MWth 해양용 용융염 원자로 개념설계 보고서 PDF Generator

Generates a professional ~100+ page A4 PDF report using fpdf2.
Chapter content is imported from separate modules in report/chapters/.

Usage:
    python report/generate_pdf.py
"""

import os
import sys
import importlib
from fpdf import FPDF

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FONT_PATH = "/System/Library/Fonts/AppleSDGothicNeo.ttc"
FONT_FAMILY = "AppleSD"

# Colors (R, G, B)
COLOR_NAVY = (26, 54, 93)          # #1a365d - headings
COLOR_DARK_GRAY = (45, 55, 72)     # #2d3748 - body text
COLOR_LIGHT_BLUE = (235, 248, 255) # #ebf8ff - table headers
COLOR_WHITE = (255, 255, 255)
COLOR_BLACK = (0, 0, 0)
COLOR_TABLE_ALT = (248, 250, 252)  # #f8fafc - alternating table rows
COLOR_NOTE_BG = (255, 251, 235)    # #fffbeb - note background
COLOR_NOTE_BORDER = (217, 175, 66) # #d9af42 - note border
COLOR_EQUATION_BG = (247, 250, 252)  # #f7fafc
COLOR_COVER_BOX = (237, 242, 247)  # #edf2f7
COLOR_LIGHT_NAVY = (44, 82, 130)   # #2c5282 - lighter navy for accents
COLOR_MEDIUM_GRAY = (113, 128, 150) # #718096

# Font sizes
SIZE_COVER_TITLE = 26
SIZE_COVER_SUBTITLE = 13
SIZE_CHAPTER_TITLE = 18
SIZE_SECTION_TITLE = 14
SIZE_SUBSECTION_TITLE = 12
SIZE_BODY = 10
SIZE_TABLE = 8
SIZE_TABLE_HEADER = 8.5
SIZE_HEADER = 8
SIZE_FOOTER = 8
SIZE_EQUATION = 10
SIZE_NOTE = 9
SIZE_TOC = 10
SIZE_CAPTION = 8.5

# Margins
MARGIN_LEFT = 25
MARGIN_RIGHT = 20
MARGIN_TOP = 25
MARGIN_BOTTOM = 25

# Line heights
LINE_HEIGHT_BODY = 7      # ~1.5x for 10pt
LINE_HEIGHT_TABLE = 6
LINE_HEIGHT_LIST = 6.5

# Page dimensions (A4)
PAGE_WIDTH = 210
PAGE_HEIGHT = 297
CONTENT_WIDTH = PAGE_WIDTH - MARGIN_LEFT - MARGIN_RIGHT  # 165mm


# ---------------------------------------------------------------------------
# Chapter registry
# ---------------------------------------------------------------------------

CHAPTERS = [
    ("ch01", 1, "서론 및 설계 기준"),
    ("ch02", 2, "노심 핵설계"),
    ("ch03", 3, "열수력 해석"),
    ("ch04", 4, "열교환기 설계"),
    ("ch05", 5, "구조 건전성 평가"),
    ("ch06", 6, "안전 해석"),
    ("ch07", 7, "차폐 설계"),
    ("ch08", 8, "선박 통합"),
    ("ch09", 9, "결론 및 향후 과제"),
    ("references", None, "참고문헌 및 부록"),
]


# ---------------------------------------------------------------------------
# MSRReport class
# ---------------------------------------------------------------------------

class MSRReport(FPDF):
    """
    Custom FPDF subclass for generating the 40 MWth Marine MSR
    conceptual design report with Korean language support.
    """

    def __init__(self):
        super().__init__(orientation="P", unit="mm", format="A4")
        self.set_margins(left=MARGIN_LEFT, top=MARGIN_TOP, right=MARGIN_RIGHT)
        self.set_auto_page_break(auto=True, margin=MARGIN_BOTTOM)

        # Register Korean fonts
        self.add_font(FONT_FAMILY, "", FONT_PATH)
        self.add_font(FONT_FAMILY, "B", FONT_PATH)

        # Internal state for TOC
        self._toc_entries = []      # (level, number, title, page)
        self._is_cover = False
        self._is_toc = False
        self._current_chapter = ""

    # ------------------------------------------------------------------
    # Header / Footer
    # ------------------------------------------------------------------

    def header(self):
        """Page header: report title (small) on non-cover pages."""
        if self._is_cover:
            return
        self.set_font(FONT_FAMILY, "", SIZE_HEADER)
        self.set_text_color(*COLOR_MEDIUM_GRAY)
        self.set_y(10)
        self.cell(
            w=CONTENT_WIDTH, h=5,
            text="40 MWth 해양용 용융염 원자로 개념설계 보고서",
            align="L",
        )
        # Page number on right
        self.cell(
            w=0, h=5,
            text=str(self.page_no()),
            align="R",
            new_x="LMARGIN", new_y="NEXT",
        )
        # Thin line under header
        self.set_draw_color(*COLOR_MEDIUM_GRAY)
        self.set_line_width(0.2)
        y = self.get_y() + 1
        self.line(MARGIN_LEFT, y, PAGE_WIDTH - MARGIN_RIGHT, y)
        self.set_y(MARGIN_TOP)

    def footer(self):
        """Page footer: report title centered."""
        if self._is_cover:
            return
        self.set_y(-15)
        self.set_font(FONT_FAMILY, "", SIZE_FOOTER)
        self.set_text_color(*COLOR_MEDIUM_GRAY)
        # Thin line above footer
        self.set_draw_color(*COLOR_MEDIUM_GRAY)
        self.set_line_width(0.2)
        y = self.get_y() - 2
        self.line(MARGIN_LEFT, y, PAGE_WIDTH - MARGIN_RIGHT, y)
        self.cell(
            w=0, h=10,
            text="40 MWth 해양용 용융염 원자로 개념설계 보고서",
            align="C",
        )

    # ------------------------------------------------------------------
    # Cover Page
    # ------------------------------------------------------------------

    def add_cover_page(self):
        """Generate the cover page with title, subtitle, and key specs."""
        self._is_cover = True
        self.add_page()

        # Top decorative bar
        self.set_fill_color(*COLOR_NAVY)
        self.rect(0, 0, PAGE_WIDTH, 8, style="F")

        # Vertical accent line on left
        self.set_fill_color(*COLOR_LIGHT_NAVY)
        self.rect(MARGIN_LEFT - 5, 35, 2, 180, style="F")

        # Title block - positioned in upper third
        self.set_y(55)
        self.set_font(FONT_FAMILY, "B", SIZE_COVER_TITLE)
        self.set_text_color(*COLOR_NAVY)
        self.multi_cell(
            w=CONTENT_WIDTH, h=14,
            text="40 MWth 해양용 용융염 원자로\n개념설계 보고서",
            align="L",
        )
        self.ln(4)

        # Subtitle
        self.set_font(FONT_FAMILY, "", SIZE_COVER_SUBTITLE)
        self.set_text_color(*COLOR_LIGHT_NAVY)
        self.multi_cell(
            w=CONTENT_WIDTH, h=7,
            text="Conceptual Design Report\nfor a 40 MWth Marine Molten Salt Reactor",
            align="L",
        )
        self.ln(8)

        # Decorative line
        self.set_draw_color(*COLOR_NAVY)
        self.set_line_width(1.0)
        y = self.get_y()
        self.line(MARGIN_LEFT, y, MARGIN_LEFT + 60, y)
        self.ln(10)

        # Key specifications box
        box_x = MARGIN_LEFT
        box_y = self.get_y()
        box_w = CONTENT_WIDTH
        box_h = 82

        # Box background
        self.set_fill_color(*COLOR_COVER_BOX)
        self.rect(box_x, box_y, box_w, box_h, style="F")

        # Box left accent
        self.set_fill_color(*COLOR_NAVY)
        self.rect(box_x, box_y, 3, box_h, style="F")

        # Box content
        self.set_y(box_y + 6)
        self.set_x(box_x + 10)
        self.set_font(FONT_FAMILY, "B", 11)
        self.set_text_color(*COLOR_NAVY)
        self.cell(w=box_w - 15, h=7, text="주요 설계 제원")
        self.ln(9)

        specs = [
            ("대상 선박", "6,000 TEU 파나막스급 컨테이너선"),
            ("열출력 / 전기출력", "40 MWth / ~16 MWe"),
            ("냉각재/연료", "FLiBe (LiF-BeF\u2082) + UF\u2084"),
            ("감속재", "핵급 흑연 (IG-110)"),
            ("구조재", "Hastelloy-N (UNS N10003)"),
            ("동력변환", "초임계 CO\u2082 브레이턴 사이클 (효율 ~40%)"),
            ("설계 수명", "20년 (용량계수 85%)"),
        ]

        self.set_font(FONT_FAMILY, "", SIZE_BODY)
        for label, value in specs:
            self.set_x(box_x + 10)
            self.set_font(FONT_FAMILY, "B", SIZE_BODY)
            self.set_text_color(*COLOR_DARK_GRAY)
            self.cell(w=50, h=8, text=label)
            self.set_font(FONT_FAMILY, "", SIZE_BODY)
            self.set_text_color(*COLOR_DARK_GRAY)
            self.cell(w=box_w - 65, h=8, text=value)
            self.ln(8)

        # Date and organization at bottom
        self.set_y(240)
        self.set_font(FONT_FAMILY, "", 12)
        self.set_text_color(*COLOR_NAVY)
        self.cell(w=CONTENT_WIDTH, h=8, text="2026년 2월", align="C")
        self.ln(10)

        # Bottom decorative bar
        self.set_fill_color(*COLOR_NAVY)
        self.rect(0, PAGE_HEIGHT - 8, PAGE_WIDTH, 8, style="F")

        self._is_cover = False

    # ------------------------------------------------------------------
    # Table of Contents
    # ------------------------------------------------------------------

    def add_toc_page(self):
        """Generate table of contents from registered entries."""
        self._is_toc = True
        self.add_page()

        # Title
        self.set_font(FONT_FAMILY, "B", SIZE_CHAPTER_TITLE)
        self.set_text_color(*COLOR_NAVY)
        self.cell(w=0, h=12, text="목  차", align="C")
        self.ln(5)

        # Decorative line
        self.set_draw_color(*COLOR_NAVY)
        self.set_line_width(0.8)
        y = self.get_y()
        center = PAGE_WIDTH / 2
        self.line(center - 30, y, center + 30, y)
        self.ln(10)

        self._is_toc = False

    def write_toc_entries(self):
        """Write TOC entries. Call after all chapters are written."""
        # We need to go back and fill the TOC page
        # Since fpdf2 doesn't support page insertion easily,
        # we build the TOC after collecting entries, then write it.
        pass  # TOC is written in generate() via a two-pass approach

    def _render_toc_entries(self):
        """Render the collected TOC entries on the current page."""
        for level, num, title, page in self._toc_entries:
            if level == 0:
                # Chapter entry
                self.set_font(FONT_FAMILY, "B", SIZE_TOC + 1)
                self.set_text_color(*COLOR_NAVY)
                indent = 0
                prefix = f"제{num}장  "
            elif level == 1:
                # Section entry
                self.set_font(FONT_FAMILY, "", SIZE_TOC)
                self.set_text_color(*COLOR_DARK_GRAY)
                indent = 10
                prefix = ""
            else:
                # Subsection entry
                self.set_font(FONT_FAMILY, "", SIZE_TOC - 1)
                self.set_text_color(*COLOR_MEDIUM_GRAY)
                indent = 20
                prefix = ""

            self.set_x(MARGIN_LEFT + indent)
            entry_text = f"{prefix}{title}"
            page_text = str(page)

            # Calculate widths
            text_w = self.get_string_width(entry_text)
            page_w = self.get_string_width(page_text)
            available_w = CONTENT_WIDTH - indent - page_w - 5

            # Entry text
            self.cell(w=available_w, h=7, text=entry_text)

            # Dotted leader
            dots_x_start = MARGIN_LEFT + indent + text_w + 2
            dots_x_end = PAGE_WIDTH - MARGIN_RIGHT - page_w - 3
            if dots_x_start < dots_x_end:
                self.set_font(FONT_FAMILY, "", 7)
                self.set_text_color(*COLOR_MEDIUM_GRAY)
                dot_str = " . " * int((dots_x_end - dots_x_start) / self.get_string_width(" . "))
                # We just use space to right-align page number
                pass

            # Page number right-aligned
            self.set_x(PAGE_WIDTH - MARGIN_RIGHT - page_w - 2)
            if level == 0:
                self.set_font(FONT_FAMILY, "B", SIZE_TOC + 1)
                self.set_text_color(*COLOR_NAVY)
            elif level == 1:
                self.set_font(FONT_FAMILY, "", SIZE_TOC)
                self.set_text_color(*COLOR_DARK_GRAY)
            else:
                self.set_font(FONT_FAMILY, "", SIZE_TOC - 1)
                self.set_text_color(*COLOR_MEDIUM_GRAY)
            self.cell(w=page_w + 2, h=7, text=page_text, align="R")
            self.ln(7 if level == 0 else 6)

            # Extra spacing after chapter entries
            if level == 0:
                self.ln(1)

            # Check page break
            if self.get_y() > PAGE_HEIGHT - MARGIN_BOTTOM - 10:
                self.add_page()

    # ------------------------------------------------------------------
    # Chapter / Section Headings
    # ------------------------------------------------------------------

    def chapter_title(self, num, title):
        """
        Chapter heading with decorative elements.
        Always starts on a new page.
        """
        self.add_page()
        self._current_chapter = f"제{num}장"

        # Register TOC entry
        self._toc_entries.append((0, num, title, self.page_no()))

        # Chapter number
        self.set_font(FONT_FAMILY, "B", 12)
        self.set_text_color(*COLOR_LIGHT_NAVY)
        self.cell(w=0, h=8, text=f"제 {num} 장")
        self.ln(10)

        # Chapter title
        self.set_font(FONT_FAMILY, "B", SIZE_CHAPTER_TITLE)
        self.set_text_color(*COLOR_NAVY)
        self.multi_cell(w=CONTENT_WIDTH, h=10, text=title)
        self.ln(2)

        # Decorative line under title
        self.set_draw_color(*COLOR_NAVY)
        self.set_line_width(1.0)
        y = self.get_y()
        self.line(MARGIN_LEFT, y, MARGIN_LEFT + 40, y)

        # Thinner continuation line
        self.set_line_width(0.3)
        self.line(MARGIN_LEFT + 42, y, PAGE_WIDTH - MARGIN_RIGHT, y)
        self.ln(10)

        # Reset text color
        self.set_text_color(*COLOR_DARK_GRAY)

    def section_title(self, text, toc=True):
        """
        Section heading (e.g., "2.1 노심 기하학적 설계").

        Args:
            text: Section title text (e.g., "2.1 노심 기하학적 설계")
            toc: Whether to register in table of contents
        """
        # Check if we need a page break (at least 30mm needed)
        if self.get_y() > PAGE_HEIGHT - MARGIN_BOTTOM - 30:
            self.add_page()

        self.ln(6)

        if toc:
            self._toc_entries.append((1, None, text, self.page_no()))

        # Section title with left accent bar
        y = self.get_y()
        self.set_fill_color(*COLOR_NAVY)
        self.rect(MARGIN_LEFT, y, 2.5, 8, style="F")

        self.set_x(MARGIN_LEFT + 5)
        self.set_font(FONT_FAMILY, "B", SIZE_SECTION_TITLE)
        self.set_text_color(*COLOR_NAVY)
        self.cell(w=CONTENT_WIDTH - 5, h=8, text=text)
        self.ln(12)

        # Reset text color
        self.set_text_color(*COLOR_DARK_GRAY)

    def subsection_title(self, text, toc=False):
        """
        Subsection heading (e.g., "2.1.1 연료채널 배열").

        Args:
            text: Subsection title text
            toc: Whether to register in table of contents
        """
        if self.get_y() > PAGE_HEIGHT - MARGIN_BOTTOM - 25:
            self.add_page()

        self.ln(4)

        if toc:
            self._toc_entries.append((2, None, text, self.page_no()))

        self.set_font(FONT_FAMILY, "B", SIZE_SUBSECTION_TITLE)
        self.set_text_color(*COLOR_LIGHT_NAVY)
        self.cell(w=CONTENT_WIDTH, h=7, text=text)
        self.ln(9)

        # Reset text color
        self.set_text_color(*COLOR_DARK_GRAY)

    # ------------------------------------------------------------------
    # Body Content
    # ------------------------------------------------------------------

    def body_text(self, text):
        """
        Body paragraph with proper line spacing (1.5x).
        Handles automatic line wrapping and page breaks.
        """
        self.set_font(FONT_FAMILY, "", SIZE_BODY)
        self.set_text_color(*COLOR_DARK_GRAY)
        self.multi_cell(
            w=CONTENT_WIDTH,
            h=LINE_HEIGHT_BODY,
            text=text,
        )
        self.ln(3)

    def body_text_bold(self, text):
        """Bold body text for emphasis."""
        self.set_font(FONT_FAMILY, "B", SIZE_BODY)
        self.set_text_color(*COLOR_DARK_GRAY)
        self.multi_cell(
            w=CONTENT_WIDTH,
            h=LINE_HEIGHT_BODY,
            text=text,
        )
        self.ln(3)

    # ------------------------------------------------------------------
    # Tables
    # ------------------------------------------------------------------

    def add_table(self, headers, rows, col_widths=None, title=None):
        """
        Formatted table with alternating row colors.

        Args:
            headers: List of header strings.
            rows: List of lists (each inner list = one row of cell strings).
            col_widths: Optional list of column widths in mm. If None,
                        widths are distributed equally.
            title: Optional table title/caption displayed above the table.
        """
        n_cols = len(headers)
        if col_widths is None:
            col_widths = [CONTENT_WIDTH / n_cols] * n_cols

        # Ensure widths sum to content width
        total = sum(col_widths)
        if abs(total - CONTENT_WIDTH) > 1:
            scale = CONTENT_WIDTH / total
            col_widths = [w * scale for w in col_widths]

        # Table title / caption
        if title:
            self.ln(2)
            self.set_font(FONT_FAMILY, "B", SIZE_CAPTION + 1)
            self.set_text_color(*COLOR_NAVY)
            self.cell(w=CONTENT_WIDTH, h=6, text=title)
            self.ln(5)

        # Check if header + at least 2 rows fit on current page
        needed_h = LINE_HEIGHT_TABLE * (1 + min(len(rows), 2)) + 5
        if self.get_y() + needed_h > PAGE_HEIGHT - MARGIN_BOTTOM:
            self.add_page()

        # Header row
        self.set_font(FONT_FAMILY, "B", SIZE_TABLE_HEADER)
        self.set_fill_color(*COLOR_LIGHT_BLUE)
        self.set_text_color(*COLOR_NAVY)
        self.set_draw_color(*COLOR_LIGHT_NAVY)
        self.set_line_width(0.3)

        for i, hdr in enumerate(headers):
            self.cell(
                w=col_widths[i], h=LINE_HEIGHT_TABLE + 1,
                text=f" {hdr}", border=1, fill=True,
            )
        self.ln(LINE_HEIGHT_TABLE + 1)

        # Data rows
        self.set_font(FONT_FAMILY, "", SIZE_TABLE)
        self.set_text_color(*COLOR_DARK_GRAY)
        self.set_draw_color(200, 210, 220)

        for row_idx, row in enumerate(rows):
            # Check page break before each row
            if self.get_y() + LINE_HEIGHT_TABLE > PAGE_HEIGHT - MARGIN_BOTTOM:
                self.add_page()
                # Re-draw header on new page
                self.set_font(FONT_FAMILY, "B", SIZE_TABLE_HEADER)
                self.set_fill_color(*COLOR_LIGHT_BLUE)
                self.set_text_color(*COLOR_NAVY)
                self.set_draw_color(*COLOR_LIGHT_NAVY)
                self.set_line_width(0.3)
                for i, hdr in enumerate(headers):
                    self.cell(
                        w=col_widths[i], h=LINE_HEIGHT_TABLE + 1,
                        text=f" {hdr}", border=1, fill=True,
                    )
                self.ln(LINE_HEIGHT_TABLE + 1)
                self.set_font(FONT_FAMILY, "", SIZE_TABLE)
                self.set_text_color(*COLOR_DARK_GRAY)
                self.set_draw_color(200, 210, 220)

            # Alternating row background
            if row_idx % 2 == 1:
                self.set_fill_color(*COLOR_TABLE_ALT)
                fill = True
            else:
                fill = False

            for i, cell_text in enumerate(row):
                self.cell(
                    w=col_widths[i], h=LINE_HEIGHT_TABLE,
                    text=f" {cell_text}", border="LR",
                    fill=fill,
                )
            self.ln(LINE_HEIGHT_TABLE)

        # Bottom border of table
        self.set_draw_color(*COLOR_LIGHT_NAVY)
        self.set_line_width(0.3)
        y = self.get_y()
        self.line(MARGIN_LEFT, y, MARGIN_LEFT + sum(col_widths), y)
        self.ln(5)

        # Reset
        self.set_text_color(*COLOR_DARK_GRAY)

    # ------------------------------------------------------------------
    # Equations
    # ------------------------------------------------------------------

    def add_equation(self, text, label=None):
        """
        Centered equation using Unicode math symbols.

        Args:
            text: Equation text with Unicode math symbols.
            label: Optional equation label (e.g., "(2.1)")
        """
        if self.get_y() > PAGE_HEIGHT - MARGIN_BOTTOM - 15:
            self.add_page()

        self.ln(3)

        # Background stripe
        y = self.get_y()
        self.set_fill_color(*COLOR_EQUATION_BG)
        self.rect(MARGIN_LEFT, y, CONTENT_WIDTH, 10, style="F")

        # Equation text (centered)
        self.set_font(FONT_FAMILY, "", SIZE_EQUATION)
        self.set_text_color(*COLOR_DARK_GRAY)

        if label:
            # Equation text centered, label right-aligned
            eq_w = CONTENT_WIDTH - 25
            self.cell(w=eq_w, h=10, text=text, align="C")
            self.set_font(FONT_FAMILY, "", SIZE_EQUATION - 1)
            self.set_text_color(*COLOR_MEDIUM_GRAY)
            self.cell(w=25, h=10, text=label, align="R")
        else:
            self.cell(w=CONTENT_WIDTH, h=10, text=text, align="C")

        self.ln(13)

        # Reset
        self.set_text_color(*COLOR_DARK_GRAY)

    # ------------------------------------------------------------------
    # Notes / Callout Boxes
    # ------------------------------------------------------------------

    def add_note(self, text, note_type="note"):
        """
        Highlighted note/callout box.

        Args:
            text: Note content text.
            note_type: "note", "warning", or "info"
        """
        if self.get_y() > PAGE_HEIGHT - MARGIN_BOTTOM - 25:
            self.add_page()

        self.ln(3)

        # Choose colors based on type
        if note_type == "warning":
            bg_color = (255, 245, 245)   # light red
            border_color = (197, 48, 48)  # red
            label = "주의"
        elif note_type == "info":
            bg_color = (235, 248, 255)    # light blue
            border_color = COLOR_LIGHT_NAVY
            label = "참고"
        else:
            bg_color = COLOR_NOTE_BG
            border_color = COLOR_NOTE_BORDER
            label = "참고"

        # Calculate text height
        self.set_font(FONT_FAMILY, "", SIZE_NOTE)
        # Estimate needed height (approximate)
        text_width = CONTENT_WIDTH - 15
        chars_per_line = text_width / (self.get_string_width("가") or 3)
        n_lines = max(1, len(text) / max(1, chars_per_line))
        box_h = max(14, n_lines * 5.5 + 12)

        y = self.get_y()

        # Background
        self.set_fill_color(*bg_color)
        self.rect(MARGIN_LEFT, y, CONTENT_WIDTH, box_h, style="F")

        # Left accent bar
        self.set_fill_color(*border_color)
        self.rect(MARGIN_LEFT, y, 3, box_h, style="F")

        # Label
        self.set_xy(MARGIN_LEFT + 7, y + 3)
        self.set_font(FONT_FAMILY, "B", SIZE_NOTE)
        self.set_text_color(*border_color)
        self.cell(w=30, h=5, text=f"\u25B6 {label}")
        self.ln(6)

        # Note text
        self.set_x(MARGIN_LEFT + 7)
        self.set_font(FONT_FAMILY, "", SIZE_NOTE)
        self.set_text_color(*COLOR_DARK_GRAY)
        self.multi_cell(w=text_width, h=5, text=text)

        self.set_y(y + box_h + 4)

        # Reset
        self.set_text_color(*COLOR_DARK_GRAY)

    # ------------------------------------------------------------------
    # Lists
    # ------------------------------------------------------------------

    def add_bullet_list(self, items):
        """
        Bulleted list.

        Args:
            items: List of strings (each string = one bullet point).
                   Can also be a list of (text, [sub_items]) tuples.
        """
        self.set_font(FONT_FAMILY, "", SIZE_BODY)
        self.set_text_color(*COLOR_DARK_GRAY)

        for item in items:
            if isinstance(item, tuple):
                text, sub_items = item
            else:
                text = item
                sub_items = []

            # Check page break
            if self.get_y() > PAGE_HEIGHT - MARGIN_BOTTOM - 10:
                self.add_page()

            # Bullet character
            self.set_x(MARGIN_LEFT + 3)
            self.set_font(FONT_FAMILY, "", SIZE_BODY)
            self.cell(w=5, h=LINE_HEIGHT_LIST, text="\u2022")
            self.multi_cell(
                w=CONTENT_WIDTH - 8,
                h=LINE_HEIGHT_LIST,
                text=text,
            )

            # Sub-items
            for sub in sub_items:
                if self.get_y() > PAGE_HEIGHT - MARGIN_BOTTOM - 10:
                    self.add_page()
                self.set_x(MARGIN_LEFT + 10)
                self.set_font(FONT_FAMILY, "", SIZE_BODY - 1)
                self.cell(w=5, h=LINE_HEIGHT_LIST, text="\u2013")
                self.multi_cell(
                    w=CONTENT_WIDTH - 15,
                    h=LINE_HEIGHT_LIST,
                    text=sub,
                )

        self.ln(2)

    def add_numbered_list(self, items, start=1):
        """
        Numbered list.

        Args:
            items: List of strings.
            start: Starting number (default 1).
        """
        self.set_font(FONT_FAMILY, "", SIZE_BODY)
        self.set_text_color(*COLOR_DARK_GRAY)

        for i, item in enumerate(items, start=start):
            if self.get_y() > PAGE_HEIGHT - MARGIN_BOTTOM - 10:
                self.add_page()

            self.set_x(MARGIN_LEFT + 3)
            self.set_font(FONT_FAMILY, "", SIZE_BODY)
            num_text = f"{i}."
            self.cell(w=8, h=LINE_HEIGHT_LIST, text=num_text)
            self.multi_cell(
                w=CONTENT_WIDTH - 11,
                h=LINE_HEIGHT_LIST,
                text=item,
            )

        self.ln(2)

    # ------------------------------------------------------------------
    # Additional helpers
    # ------------------------------------------------------------------

    def add_figure_placeholder(self, caption, height=60):
        """
        Add a placeholder box for a figure.

        Args:
            caption: Figure caption text.
            height: Height of the placeholder box in mm.
        """
        if self.get_y() + height + 15 > PAGE_HEIGHT - MARGIN_BOTTOM:
            self.add_page()

        self.ln(3)
        y = self.get_y()

        # Dashed border placeholder
        self.set_draw_color(*COLOR_MEDIUM_GRAY)
        self.set_line_width(0.3)
        self.set_fill_color(250, 250, 252)
        self.rect(MARGIN_LEFT + 10, y, CONTENT_WIDTH - 20, height, style="FD")

        # Centered placeholder text
        self.set_font(FONT_FAMILY, "", 9)
        self.set_text_color(*COLOR_MEDIUM_GRAY)
        self.set_xy(MARGIN_LEFT + 10, y + height / 2 - 4)
        self.cell(
            w=CONTENT_WIDTH - 20, h=8,
            text="[Figure Placeholder]",
            align="C",
        )

        self.set_y(y + height + 3)

        # Caption
        self.set_font(FONT_FAMILY, "", SIZE_CAPTION)
        self.set_text_color(*COLOR_DARK_GRAY)
        self.multi_cell(w=CONTENT_WIDTH, h=5, text=caption, align="C")
        self.ln(4)

        # Reset
        self.set_text_color(*COLOR_DARK_GRAY)

    def add_page_break(self):
        """Force a page break."""
        self.add_page()

    def add_spacing(self, mm=5):
        """Add vertical spacing."""
        self.ln(mm)

    # ------------------------------------------------------------------
    # Two-pass TOC Generation
    # ------------------------------------------------------------------

    def generate(self):
        """
        Generate the complete report with two-pass TOC.

        Pass 1: Generate all content and collect TOC entries + page numbers.
        Pass 2: Rebuild the PDF with a proper TOC page inserted after cover.
        """
        # ---- Pass 1: Build content, collect TOC entries ----
        self._build_content()

        # ---- Pass 2: Rebuild with TOC ----
        # Store collected TOC entries
        toc_entries = list(self._toc_entries)

        # Create fresh instance for pass 2
        self.__init__()
        self._toc_entries = toc_entries

        # We need to adjust page numbers: TOC adds pages
        # First, count how many TOC pages we need
        toc_page_count = self._estimate_toc_pages(toc_entries)

        # Adjust all TOC page numbers
        adjusted_entries = []
        for level, num, title, page in toc_entries:
            adjusted_entries.append((level, num, title, page + toc_page_count))
        self._toc_entries = adjusted_entries

        # Now build the final PDF
        self._build_content(with_toc=True)

    def _estimate_toc_pages(self, entries):
        """Estimate how many pages the TOC will need."""
        lines = 0
        for level, _, _, _ in entries:
            lines += 1
            if level == 0:
                lines += 0.3  # extra spacing after chapter
        lines_per_page = 35  # conservative estimate
        return max(1, int(lines / lines_per_page) + 1)

    def _build_content(self, with_toc=False):
        """Build all PDF content."""
        # Cover page
        self.add_cover_page()

        # TOC page (pass 2 only)
        if with_toc:
            self.add_toc_page()
            self._render_toc_entries()

        # Chapters
        self._write_chapters()

    def _write_chapters(self):
        """Import and execute each chapter module."""
        # Determine the base directory for chapter imports
        base_dir = os.path.dirname(os.path.abspath(__file__))
        chapters_dir = os.path.join(base_dir, "chapters")

        # Ensure chapters dir is importable
        if base_dir not in sys.path:
            sys.path.insert(0, base_dir)

        for module_name, ch_num, ch_title in CHAPTERS:
            module_path = f"chapters.{module_name}"
            try:
                mod = importlib.import_module(module_path)
                # Reload in case of pass 2
                importlib.reload(mod)
                if hasattr(mod, "write_chapter"):
                    mod.write_chapter(self)
                else:
                    # Module exists but no write_chapter function
                    self._write_placeholder_chapter(ch_num, ch_title)
            except ImportError:
                # Chapter module doesn't exist yet - write placeholder
                self._write_placeholder_chapter(ch_num, ch_title)

    def _write_placeholder_chapter(self, num, title):
        """Write a placeholder chapter when the real module isn't available."""
        if num is None:
            # Non-numbered section (e.g., references, appendix)
            self.add_page()
            self.set_font(FONT_FAMILY, "B", SIZE_CHAPTER_TITLE)
            self.set_text_color(*COLOR_NAVY)
            self.multi_cell(w=CONTENT_WIDTH, h=10, text=title)
            self.ln(10)
            self.set_text_color(*COLOR_DARK_GRAY)
        else:
            self.chapter_title(num, title)
        self.body_text(
            f"이 장의 내용은 chapters/ch{num:02d}.py 모듈에서 제공됩니다. "
            f"해당 모듈의 write_chapter(pdf) 함수를 구현하면 "
            f"자동으로 이 자리표시자를 대체합니다."
        )
        self.add_note(
            f"ch{num:02d}.py 모듈을 생성하고 write_chapter(pdf) 함수를 구현하세요. "
            f"pdf 인스턴스의 helper 메서드들 (body_text, add_table, add_equation 등)을 "
            f"사용하여 내용을 작성할 수 있습니다.",
            note_type="info",
        )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main():
    """Generate the MSR design report PDF."""
    # Ensure output directory exists
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, "40MWth_Marine_MSR_Design_Report.pdf")

    print("=" * 60)
    print("40 MWth 해양용 용융염 원자로 개념설계 보고서")
    print("PDF Report Generator")
    print("=" * 60)

    report = MSRReport()

    print("\n[1/3] Building content (Pass 1)...")
    print("      Collecting TOC entries and page numbers...")

    report.generate()

    print("[2/3] Content built with TOC.")
    print(f"      Total pages: {report.pages_count}")
    print(f"      TOC entries: {len(report._toc_entries)}")

    print(f"[3/3] Writing PDF to: {output_path}")
    report.output(output_path)

    file_size = os.path.getsize(output_path)
    print(f"\n  Output: {output_path}")
    print(f"  Size:   {file_size / 1024:.1f} KB")
    print(f"  Pages:  {report.pages_count}")
    print("  Done.")


if __name__ == "__main__":
    main()
