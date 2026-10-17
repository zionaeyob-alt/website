// Build report/final_report.docx from report/final_report.md.
//
// Usage (from the project root):  node scripts/build_report_docx.js
// Requires the `docx` npm package (npm install -g docx). The Markdown file stays the single
// source of truth; this script only handles the constructs used in the report: headings,
// paragraphs, **bold** / *italic* / `code`, tables, fenced code, numbered lists, figures with
// an italic caption line, and the reference list.

const fs = require("fs");
const path = require("path");
let docx;
try { docx = require("docx"); } catch { docx = require("/opt/node22/lib/node_modules/docx"); }
const {
  AlignmentType, BorderStyle, Document, Footer, Header, HeadingLevel, ImageRun, LevelFormat,
  Packer, PageBreak, PageNumber, Paragraph, ShadingType, Table, TableCell, TableRow, TabStopType,
  TextRun, VerticalAlign, WidthType,
} = docx;

const ROOT = path.resolve(__dirname, "..");
const MD_PATH = path.join(ROOT, "report", "final_report.md");
const OUT_PATH = path.join(ROOT, "report", "final_report.docx");

// US Letter, 1-inch margins -> 6.5-inch (9360 DXA) text width.
const PAGE = { width: 12240, height: 15840, margin: 1440 };
const TEXT_W = PAGE.width - 2 * PAGE.margin;
const FONT = "Calibri";
const MONO = "Consolas";
const ACCENT = "1F4E79";
const RULE = "BFBFBF";
const HEADER_FILL = "DCE6F1";
const CODE_FILL = "F3F3F3";
// Display width per figure (inches), chosen so text inside each chart stays legible.
const FIG_WIDTH_IN = { "01": 4.2, "02": 4.0, "03": 5.6, "04": 6.2, "05": 3.9, "06": 6.5, "07": 4.3, "08": 5.2 };

// ----------------------------------------------------------------------------- inline parsing
function inlineRuns(text, base = {}) {
  const runs = [];
  const re = /(\*\*[^*]+\*\*|\*[^*\s][^*]*\*|`[^`]+`)/g;
  let last = 0, m;
  while ((m = re.exec(text)) !== null) {
    if (m.index > last) runs.push(new TextRun({ text: text.slice(last, m.index), ...base }));
    const tok = m[0];
    if (tok.startsWith("**")) {
      // Bold may contain inline code.
      inlineRuns(tok.slice(2, -2), { ...base, bold: true }).forEach(r => runs.push(r));
    } else if (tok.startsWith("`")) {
      runs.push(new TextRun({ text: tok.slice(1, -1), ...base, font: MONO, size: (base.size || 22) - 2 }));
    } else {
      runs.push(new TextRun({ text: tok.slice(1, -1), ...base, italics: true }));
    }
    last = m.index + tok.length;
  }
  if (last < text.length) runs.push(new TextRun({ text: text.slice(last), ...base }));
  return runs;
}

const para = (text, opts = {}, runOpts = {}) =>
  new Paragraph({ children: inlineRuns(text, runOpts), spacing: { after: 100 }, ...opts });

// ----------------------------------------------------------------------------- block builders
function pngSize(file) {
  const buf = fs.readFileSync(file);
  return { w: buf.readUInt32BE(16), h: buf.readUInt32BE(20), buf };
}

function figure(imgPath, caption) {
  const abs = path.resolve(path.dirname(MD_PATH), imgPath);
  const { w, h, buf } = pngSize(abs);
  const key = path.basename(abs).slice(0, 2);
  const widthPx = (FIG_WIDTH_IN[key] || 6.0) * 96;
  const heightPx = widthPx * (h / w);
  const label = caption.match(/^(Figure \d+\.)\s*(.*)$/);
  return [
    new Paragraph({
      alignment: AlignmentType.CENTER, keepNext: true, spacing: { before: 120, after: 40 },
      children: [new ImageRun({
        type: "png", data: buf, transformation: { width: widthPx, height: heightPx },
        altText: { title: label ? label[1] : "Figure", description: caption, name: path.basename(abs) },
      })],
    }),
    new Paragraph({
      alignment: AlignmentType.CENTER, spacing: { after: 160 },
      children: label
        ? [new TextRun({ text: label[1] + " ", bold: true, size: 19, color: "404040" }),
           ...inlineRuns(label[2], { italics: true, size: 19, color: "404040" })]
        : inlineRuns(caption, { italics: true, size: 19 }),
    }),
  ];
}

function codeBlock(lines) {
  // One shaded, bordered paragraph per line keeps monospace alignment and avoids \n.
  const border = { style: BorderStyle.SINGLE, size: 4, color: "D0D0D0", space: 4 };
  return lines.map((line, i) => new Paragraph({
    shading: { type: ShadingType.CLEAR, color: "auto", fill: CODE_FILL },
    border: {
      left: border, right: border,
      ...(i === 0 ? { top: border } : {}), ...(i === lines.length - 1 ? { bottom: border } : {}),
    },
    spacing: { before: i === 0 ? 80 : 0, after: i === lines.length - 1 ? 140 : 0 },
    keepNext: i < lines.length - 1, keepLines: true,
    children: [new TextRun({ text: line.length ? line : " ", font: MONO, size: 17 })],
  }));
}

function table(rows) {
  const cells = rows.filter(r => !/^\|\s*-/.test(r))
    .map(r => r.trim().replace(/^\||\|$/g, "").split("|").map(c => c.trim()));
  const nCols = cells[0].length;
  const fontSize = nCols >= 8 ? 16 : nCols >= 5 ? 18 : 19;
  // Column widths proportional to content length, with a floor so short headers fit.
  const lens = Array.from({ length: nCols }, (_, j) =>
    Math.max(6, ...cells.map(r => (r[j] || "").replace(/[*`]/g, "").length)));
  const capped = lens.map(l => Math.min(l, 48));
  const total = capped.reduce((a, b) => a + b, 0);
  const widths = capped.map(l => Math.floor((l / total) * TEXT_W));
  widths[widths.length - 1] += TEXT_W - widths.reduce((a, b) => a + b, 0);
  const border = { style: BorderStyle.SINGLE, size: 4, color: RULE };
  const borders = { top: border, bottom: border, left: border, right: border };
  return new Table({
    width: { size: TEXT_W, type: WidthType.DXA },
    columnWidths: widths,
    rows: cells.map((r, i) => new TableRow({
      tableHeader: i === 0, cantSplit: true,
      children: r.map((c, j) => new TableCell({
        width: { size: widths[j], type: WidthType.DXA },
        borders, verticalAlign: VerticalAlign.CENTER,
        margins: { top: 40, bottom: 40, left: 80, right: 80 },
        shading: i === 0 ? { type: ShadingType.CLEAR, color: "auto", fill: HEADER_FILL } : undefined,
        children: [new Paragraph({
          spacing: { after: 0 }, keepNext: i < cells.length - 1,
          children: inlineRuns(c, { size: fontSize, ...(i === 0 ? { bold: true } : {}) }),
        })],
      })),
    })),
  });
}

// ----------------------------------------------------------------------------- document parsing
function parse(md) {
  const lines = md.split("\n");
  const cover = { title: "", meta: [] };
  const body = [];
  let i = 0, listInstance = 0, inReferences = false;

  // Cover: "# Title", metadata lines, then "---".
  while (i < lines.length && lines[i].trim() !== "---") {
    const l = lines[i].trim();
    if (l.startsWith("# ")) cover.title = l.slice(2);
    else if (l) l.split(" · ").forEach(part => cover.meta.push(part.trim()));
    i++;
  }
  i++;

  while (i < lines.length) {
    const line = lines[i];
    const t = line.trim();
    if (!t) { i++; continue; }

    if (t.startsWith("## ")) {
      inReferences = /References/i.test(t);
      body.push(new Paragraph({
        heading: HeadingLevel.HEADING_1, keepNext: true,
        children: [new TextRun({ text: t.slice(3) })],
      }));
      i++; continue;
    }
    if (t.startsWith("### ")) {
      body.push(new Paragraph({ heading: HeadingLevel.HEADING_2, keepNext: true,
        children: [new TextRun({ text: t.slice(4) })] }));
      i++; continue;
    }
    if (t.startsWith("```")) {
      const code = [];
      i++;
      while (i < lines.length && !lines[i].trim().startsWith("```")) code.push(lines[i++]);
      i++;
      body.push(...codeBlock(code));
      continue;
    }
    if (t.startsWith("|")) {
      const rows = [];
      while (i < lines.length && lines[i].trim().startsWith("|")) rows.push(lines[i++]);
      body.push(table(rows));
      body.push(new Paragraph({ spacing: { after: 60 }, children: [] }));
      continue;
    }
    const img = t.match(/^!\[[^\]]*\]\(([^)]+)\)$/);
    if (img) {
      i++;
      while (i < lines.length && !lines[i].trim()) i++;
      let caption = "";
      const cap = lines[i] && lines[i].trim().match(/^\*(.+)\*$/);
      if (cap) {
        caption = cap[1];
        i++;
        // Captions may wrap onto following lines until the closing asterisk.
      } else if (lines[i] && lines[i].trim().startsWith("*")) {
        const parts = [];
        while (i < lines.length && lines[i].trim()) parts.push(lines[i++].trim());
        caption = parts.join(" ").replace(/^\*|\*$/g, "");
      }
      body.push(...figure(img[1], caption));
      continue;
    }
    const num = t.match(/^(\d+)\.\s+(.*)$/);
    if (num) {
      listInstance++;
      while (i < lines.length && /^\d+\.\s+/.test(lines[i].trim())) {
        const parts = [lines[i].trim().replace(/^\d+\.\s+/, "")];
        i++;
        while (i < lines.length && /^\s{2,}\S/.test(lines[i])) parts.push(lines[i++].trim());
        body.push(new Paragraph({
          numbering: { reference: "numbered", level: 0, instance: listInstance },
          spacing: { after: 60 }, children: inlineRuns(parts.join(" ")),
        }));
      }
      body.push(new Paragraph({ spacing: { after: 40 }, children: [] }));
      continue;
    }

    // Plain paragraph: join wrapped lines until a blank line or a new block.
    const parts = [];
    while (i < lines.length && lines[i].trim() &&
           !/^(##|```|\||!\[|\d+\.\s)/.test(lines[i].trim())) parts.push(lines[i++].trim());
    const text = parts.join(" ");
    body.push(inReferences
      ? para(text, { indent: { left: 720, hanging: 720 }, spacing: { after: 100 } }, { size: 20 })
      : para(text, { alignment: AlignmentType.JUSTIFIED, keepNext: text.endsWith(":") }));
  }
  return { cover, body };
}

// ----------------------------------------------------------------------------- cover page
function coverPage({ title, meta }) {
  const rows = meta.map(m => {
    const mm = m.match(/^\*\*(.+?):\*\*\s*(.*)$/);
    return mm ? [mm[1], mm[2]] : ["", m.replace(/\*\*/g, "")];
  });
  const lookup = Object.fromEntries(rows);
  const line = (label, value, size = 24) => new Paragraph({
    alignment: AlignmentType.CENTER, spacing: { after: 120 },
    children: [
      ...(label ? [new TextRun({ text: `${label}: `, bold: true, size, color: "404040" })] : []),
      ...inlineRuns(value, { size, color: "404040" }),
    ],
  });
  const course = lookup["Course"] || "";
  return [
    new Paragraph({ spacing: { before: 2600 }, children: [] }),
    new Paragraph({
      alignment: AlignmentType.CENTER, spacing: { after: 160 },
      children: [new TextRun({ text: course, size: 26, color: ACCENT, bold: true })],
    }),
    ...rows.filter(([k]) => !k).map(([, v]) => new Paragraph({
      alignment: AlignmentType.CENTER, spacing: { after: 480 },
      children: [new TextRun({ text: v, size: 26, color: ACCENT })],
    })),
    new Paragraph({
      alignment: AlignmentType.CENTER, spacing: { after: 240 },
      border: { bottom: { style: BorderStyle.SINGLE, size: 12, color: ACCENT, space: 12 } },
      children: [new TextRun({ text: title, size: 44, bold: true, color: "1A1A1A" })],
    }),
    new Paragraph({ spacing: { after: 900 }, children: [] }),
    ...["Student Name", "Student Number", "Date"].filter(k => lookup[k] !== undefined)
      .map(k => line(k, lookup[k])),
    new Paragraph({ spacing: { after: 900 }, children: [] }),
    ...["Dataset", "Tools"].filter(k => lookup[k] !== undefined).map(k => line(k, lookup[k], 20)),
    new Paragraph({ children: [new PageBreak()] }),
  ];
}

// ----------------------------------------------------------------------------- assemble
const { cover, body } = parse(fs.readFileSync(MD_PATH, "utf8"));
const shortTitle = cover.title;

const doc = new Document({
  creator: "DTO 5389 student",
  title: cover.title,
  description: "Assignment 2 – AutoML classification report",
  styles: {
    default: { document: { run: { font: FONT, size: 22 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 28, bold: true, color: ACCENT, font: FONT },
        paragraph: { spacing: { before: 240, after: 100 }, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 24, bold: true, color: ACCENT, font: FONT },
        paragraph: { spacing: { before: 160, after: 80 }, outlineLevel: 1 } },
    ],
  },
  numbering: {
    config: [{
      reference: "numbered",
      levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT,
                 style: { paragraph: { indent: { left: 360, hanging: 360 } } } }],
    }],
  },
  sections: [{
    properties: {
      page: { size: { width: PAGE.width, height: PAGE.height },
              margin: { top: PAGE.margin, bottom: PAGE.margin, left: PAGE.margin, right: PAGE.margin } },
      titlePage: true,
    },
    headers: {
      default: new Header({ children: [new Paragraph({
        alignment: AlignmentType.RIGHT,
        children: [new TextRun({ text: `DTO 5389 · Assignment 2 · ${shortTitle}`, size: 16, color: "808080" })],
      })] }),
      first: new Header({ children: [new Paragraph({ children: [] })] }),
    },
    footers: {
      default: new Footer({ children: [new Paragraph({
        alignment: AlignmentType.CENTER,
        children: [new TextRun({ children: [PageNumber.CURRENT], size: 18, color: "808080" })],
      })] }),
      first: new Footer({ children: [new Paragraph({ children: [] })] }),
    },
    children: [...coverPage(cover), ...body],
  }],
});

Packer.toBuffer(doc).then(buf => {
  fs.writeFileSync(OUT_PATH, buf);
  console.log(`wrote ${path.relative(ROOT, OUT_PATH)}`);
});
