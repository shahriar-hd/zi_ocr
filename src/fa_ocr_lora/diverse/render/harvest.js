// Harvest exact text geometry from a laid-out page.
//
// This is the file that decides whether the dataset's ground truth is real.
// The annotations are *measured*, never inferred: every line box is where
// Chromium actually drew that line, read back through Range rectangles after
// layout and after webfonts have loaded. Guessing boxes from font metrics -
// the usual shortcut - goes wrong on exactly the cases this corpus is full of:
// justified Persian, ZWNJ-joined compounds, bidi runs with Latin terms, and
// nastaliq, where the ink of one word overlaps the next.
//
// Three decisions worth knowing about.
//
// **Reading order comes from the DOM, not from coordinates.** For a
// right-to-left multi-column page no coordinate rule recovers the order
// correctly, and every heuristic fails on something. The templates emit their
// content in reading order by construction and CSS columns flow in DOM order,
// so a depth-first DOM walk *is* the reading order. It is also stable, which a
// geometric sort is not.
//
// **Lines are found per character.** A Range over one character gives that
// character's rectangle; characters are then bucketed by vertical position to
// recover the lines the browser chose. This is O(characters) Range queries -
// a few thousand on a dense page, tens of milliseconds - and it is the only
// method that gets the line *text* right as well as the line *box*, because
// the browser decides where a justified Persian line breaks.
//
// **Only `[data-ann]` subtrees are harvested.** Decoration (rules, avatars,
// figure placeholders, watermarks) carries no annotation attribute and so
// contributes no box, which is what keeps non-text out of the ground truth.

(async (options) => {
  const opts = options || {};
  const wantWords = opts.words !== false;
  // Characters whose rectangles are unreliable or meaningless. ZWNJ is
  // zero-width by definition, so it never defines a line; it is still carried
  // in the text, because it is part of the label.
  const ZERO_WIDTH = new Set(["\u200c", "\u200b", "\u200d", "\ufeff", "\u00ad"]);

  // Wait for webfonts. Measuring before they load reports boxes for the
  // fallback face, which can be a different size entirely - the resulting
  // annotation would be plausible and wrong.
  if (document.fonts && document.fonts.ready) {
    try {
      await document.fonts.ready;
    } catch (error) {
      /* fall through: a font that never resolves still has to be measured */
    }
  }
  // One more frame, so any layout triggered by font swap has settled.
  await new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve)));

  const canvas = document.getElementById("canvas") || document.body;
  const origin = canvas.getBoundingClientRect();

  // Rectangles are returned relative to the canvas, not the viewport, so a
  // page that scrolled or sits inside a padded wrapper still reports
  // coordinates that match the screenshot.
  const localise = (rect) => ({
    x0: rect.left - origin.left,
    y0: rect.top - origin.top,
    x1: rect.right - origin.left,
    y1: rect.bottom - origin.top,
  });

  const isVisible = (element) => {
    const style = window.getComputedStyle(element);
    if (style.display === "none" || style.visibility === "hidden") return false;
    if (parseFloat(style.opacity || "1") < 0.05) return false;
    const rect = element.getBoundingClientRect();
    return rect.width > 0.5 && rect.height > 0.5;
  };

  // Collect the annotated elements in DOM order. An annotated element nested
  // inside another is kept and the outer one yields only the text not claimed
  // by its children, so a <p> containing an annotated <span> does not report
  // the span's text twice.
  const annotated = Array.prototype.slice.call(document.querySelectorAll("[data-ann]"));
  const annotatedSet = new Set(annotated);

  // Gather the text nodes belonging to one annotated element, stopping at any
  // nested annotated element.
  const ownTextNodes = (root) => {
    const nodes = [];
    const walk = (node) => {
      for (let child = node.firstChild; child; child = child.nextSibling) {
        if (child.nodeType === Node.TEXT_NODE) {
          if (child.nodeValue && child.nodeValue.trim().length) nodes.push(child);
        } else if (child.nodeType === Node.ELEMENT_NODE) {
          if (annotatedSet.has(child)) continue; // belongs to that element
          if (!isVisible(child)) continue;
          walk(child);
        }
      }
    };
    walk(root);
    return nodes;
  };

  // Recover the lines of one text node by measuring each character.
  const linesOfNode = (node) => {
    const value = node.nodeValue;
    const range = document.createRange();
    const entries = [];
    for (let index = 0; index < value.length; index += 1) {
      const character = value[index];
      if (ZERO_WIDTH.has(character)) {
        entries.push({ character, rect: null });
        continue;
      }
      range.setStart(node, index);
      range.setEnd(node, index + 1);
      const rect = range.getBoundingClientRect();
      if (!rect || (rect.width === 0 && rect.height === 0)) {
        entries.push({ character, rect: null });
        continue;
      }
      entries.push({ character, rect });
    }
    range.detach && range.detach();

    // Bucket into lines. A new line starts when a character's vertical centre
    // falls outside the current line's vertical span by more than a third of
    // its height - tolerant enough for the sub-pixel jitter of mixed font
    // sizes on one line, tight enough not to merge two tight lines.
    const lines = [];
    let current = null;
    for (const entry of entries) {
      if (!entry.rect) {
        if (current) current.characters.push(entry.character);
        continue;
      }
      const rect = entry.rect;
      const centre = (rect.top + rect.bottom) / 2;
      if (
        current &&
        centre >= current.top - current.height / 3 &&
        centre <= current.bottom + current.height / 3
      ) {
        current.characters.push(entry.character);
        current.top = Math.min(current.top, rect.top);
        current.bottom = Math.max(current.bottom, rect.bottom);
        current.left = Math.min(current.left, rect.left);
        current.right = Math.max(current.right, rect.right);
        current.height = current.bottom - current.top;
        current.rects.push({ rect, character: entry.character });
      } else {
        current = {
          characters: [entry.character],
          top: rect.top,
          bottom: rect.bottom,
          left: rect.left,
          right: rect.right,
          height: rect.bottom - rect.top,
          rects: [{ rect, character: entry.character }],
        };
        lines.push(current);
      }
    }
    return lines;
  };

  // Split a measured line into word boxes on whitespace. ZWNJ does not split
  // a word: it joins one.
  const wordsOfLine = (line) => {
    const words = [];
    let current = null;
    for (const item of line.rects) {
      const character = item.character;
      if (/\s/.test(character)) {
        if (current) {
          words.push(current);
          current = null;
        }
        continue;
      }
      if (!current) {
        current = {
          text: character,
          left: item.rect.left,
          right: item.rect.right,
          top: item.rect.top,
          bottom: item.rect.bottom,
        };
      } else {
        current.text += character;
        current.left = Math.min(current.left, item.rect.left);
        current.right = Math.max(current.right, item.rect.right);
        current.top = Math.min(current.top, item.rect.top);
        current.bottom = Math.max(current.bottom, item.rect.bottom);
      }
    }
    if (current) words.push(current);
    return words;
  };

  const outLines = [];
  const outBlocks = [];
  let overflowed = false;

  for (const element of annotated) {
    if (!isVisible(element)) continue;
    const blockType = element.getAttribute("data-ann") || "other";
    const nodes = ownTextNodes(element);
    if (!nodes.length) continue;

    const blockLineIndices = [];
    const direction = window.getComputedStyle(element).direction === "ltr" ? "ltr" : "rtl";

    for (const node of nodes) {
      for (const line of linesOfNode(node)) {
        const text = line.characters.join("").trim();
        if (!text) continue;
        const box = localise({
          left: line.left,
          top: line.top,
          right: line.right,
          bottom: line.bottom,
        });
        blockLineIndices.push(outLines.length);
        const record = {
          text,
          box,
          direction,
          block_index: outBlocks.length,
        };
        if (wantWords) {
          record.words = wordsOfLine(line)
            .filter((word) => word.text.trim().length)
            .map((word) => ({
              text: word.text,
              box: localise({
                left: word.left,
                top: word.top,
                right: word.right,
                bottom: word.bottom,
              }),
            }));
        }
        outLines.push(record);
      }
    }

    if (!blockLineIndices.length) continue;
    const rect = localise(element.getBoundingClientRect());
    outBlocks.push({
      type: blockType,
      box: rect,
      line_indices: blockLineIndices,
    });
  }

  // Overflow detection. The canvas has `overflow:hidden`, so content past its
  // edge is invisible in the screenshot while still being reported by the
  // layout - exactly the case where a label would claim text the image does
  // not show. Image QC rejects on this flag.
  const canvasRect = canvas.getBoundingClientRect();
  const fits = (box) =>
    box.x1 <= canvasRect.width + 1.5 &&
    box.y1 <= canvasRect.height + 1.5 &&
    box.x0 >= -1.5 &&
    box.y0 >= -1.5;
  for (const line of outLines) {
    if (!fits(line.box)) {
      overflowed = true;
      break;
    }
  }
  if (canvas.scrollHeight > Math.ceil(canvasRect.height) + 2) overflowed = true;

  return {
    width: Math.round(canvasRect.width),
    height: Math.round(canvasRect.height),
    // The content height, so the renderer can size a fit-to-content canvas.
    content_height: Math.ceil(canvas.scrollHeight),
    lines: outLines,
    blocks: outBlocks,
    overflowed,
  };
})
