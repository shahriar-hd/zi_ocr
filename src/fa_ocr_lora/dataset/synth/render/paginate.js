/*
 * Pagination and bounding-box harvest, injected into the rendered letter.
 *
 * The whole file is one async arrow function so Playwright can hand it to
 * `page.evaluate(source, options)` and have it called with the options object.
 *
 * Why this is done in the browser rather than in Python: the only authority on
 * where a shaped, justified, bidi-resolved Persian line actually landed is the
 * engine that laid it out. `getBoundingClientRect()` asks that authority
 * directly, so a box can never disagree with the pixels beneath it.
 *
 * Sequence:
 *   1. Wait for the embedded faces, or every box is measured against a
 *      fallback font and is wrong by a few per cent.
 *   2. Drain `#flow` into cloned sheets, starting a new sheet the moment a
 *      block makes the body overflow.
 *   3. Stamp page numbers, which only exist once the page count is known.
 *   4. Read every `[data-ann-type]` node's rect relative to its own sheet.
 *
 * Returns `{sheets, boxes, overflowed}` in CSS pixels; the caller scales to
 * device pixels by the same `device_scale_factor` it screenshotted with.
 */
async (options) => {
  'use strict';

  const opts = options || {};
  const maxPages = Math.max(1, opts.maxPages || 6);
  /* Half a pixel of slack: a body whose content measures 0.02px taller than
   * its box is a rounding artefact, not an overflow. */
  const SLACK = 0.5;
  const PERSIAN_DIGITS = '۰۱۲۳۴۵۶۷۸۹';

  const toPersian = (value) =>
    String(value).replace(/[0-9]/g, (digit) => PERSIAN_DIGITS[Number(digit)]);

  const round = (value) => Math.round(value * 100) / 100;

  if (document.fonts && document.fonts.ready) {
    await document.fonts.ready;
  }

  const host = document.getElementById('sheets');
  const template = document.getElementById('sheet-template');
  const flow = document.getElementById('flow');
  if (!host || !template || !flow) {
    throw new Error('paginate: the letter template is missing a required hook');
  }

  const blocks = Array.from(flow.children);
  /* Detached before measuring so a block that never found a sheet - because
   * the page cap was reached - cannot leave a stray box behind. */
  flow.remove();

  const sheets = [];

  const addSheet = () => {
    const sheet = template.content.firstElementChild.cloneNode(true);
    host.appendChild(sheet);
    sheets.push(sheet);
    return sheet;
  };

  const bodyOf = (sheet) => sheet.querySelector('[data-region="body"]');
  const overflows = (body) => body.scrollHeight > body.clientHeight + SLACK;

  let sheet = addSheet();
  let body = bodyOf(sheet);
  let overflowed = false;

  for (const block of blocks) {
    body.appendChild(block);
    if (!overflows(body)) {
      continue;
    }
    if (body.children.length === 1) {
      /* Taller than a whole page on its own - a very long table, say. Keeping
       * it is the lesser evil: moving it to a fresh sheet would overflow that
       * one too, and dropping it would put text on no page at all. */
      overflowed = true;
      continue;
    }
    if (sheets.length >= maxPages) {
      /* Out of paper. The block stays where it is and the caller is told the
       * document was clipped, so the sample can be quarantined rather than
       * silently shipped with unlabelled ink. */
      overflowed = true;
      continue;
    }
    body.removeChild(block);
    sheet = addSheet();
    body = bodyOf(sheet);
    body.appendChild(block);
  }

  /* --------------------------------------------------------- page numbers */

  const total = sheets.length;
  sheets.forEach((node, index) => {
    node.querySelectorAll('[data-page-template]').forEach((slot) => {
      const pattern = slot.getAttribute('data-page-template') || '';
      slot.textContent = pattern
        .replace(/\{page\}/g, toPersian(index + 1))
        .replace(/\{total\}/g, toPersian(total));
    });
  });

  /* ------------------------------------------------------------- harvest */

  const boxes = [];

  sheets.forEach((node, index) => {
    const origin = node.getBoundingClientRect();
    node.querySelectorAll('[data-ann-type]').forEach((element) => {
      const rect = element.getBoundingClientRect();
      if (rect.width <= 0 || rect.height <= 0) {
        /* A field whose value turned out empty renders as a zero-area node.
         * No ink means no box, per the "never claim a box for absent content"
         * rule the layout modules also follow. */
        return;
      }
      const data = element.dataset || {};
      const extras = {};
      Object.keys(data).forEach((key) => {
        if (key === 'annType' || key === 'annId' || key === 'annContent') {
          return;
        }
        if (key.startsWith('ann')) {
          const name = key.slice(3);
          extras[name.charAt(0).toLowerCase() + name.slice(1)] = data[key];
        }
      });

      let content = null;
      if (data.annContent) {
        try {
          content = JSON.parse(data.annContent);
        } catch (error) {
          content = null;
        }
      }

      boxes.push({
        id: data.annId || '',
        type: data.annType,
        sheet: index,
        rect: [
          round(rect.left - origin.left),
          round(rect.top - origin.top),
          round(rect.right - origin.left),
          round(rect.bottom - origin.top),
        ],
        text: (element.innerText || element.textContent || '')
          .replace(/\s+/g, ' ')
          .trim(),
        attrs: extras,
        content: content,
      });
    });
  });

  return {
    sheets: sheets.map((node) => {
      const rect = node.getBoundingClientRect();
      return { width: round(rect.width), height: round(rect.height) };
    }),
    boxes: boxes,
    overflowed: overflowed,
  };
};
