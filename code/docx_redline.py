"""
Minimal OOXML redlining helpers (lxml) for editing a COPY of the manuscript.

TRACK=True  -> every change is wrapped in <w:ins>/<w:del> (visible in Word "Track Changes")
TRACK=False -> the same edits applied directly (clean copy)
"""
import copy
from lxml import etree

W = 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'
M = 'http://schemas.openxmlformats.org/officeDocument/2006/math'
XML = 'http://www.w3.org/XML/1998/namespace'
NS = {'w': W, 'm': M}


def w(tag): return f'{{{W}}}{tag}'


RPR_ORDER = ['rStyle', 'rFonts', 'b', 'bCs', 'i', 'iCs', 'caps', 'smallCaps', 'strike', 'dstrike', 'outline', 'shadow', 'emboss', 'imprint',
             'noProof', 'snapToGrid', 'vanish', 'webHidden', 'color', 'spacing', 'w', 'kern', 'position', 'sz', 'szCs', 'highlight', 'u',
             'effect', 'bdr', 'shd', 'fitText', 'vertAlign', 'rtl', 'cs', 'em', 'lang', 'eastAsianLayout', 'specVanish', 'oMath']


def sort_rPr(rp):
    """Re-order rPr children into the ECMA-376 sequence (ins/del markers first)."""
    key = lambda e: (-1 if etree.QName(e).localname in ('ins', 'del') else
                     RPR_ORDER.index(etree.QName(e).localname) if etree.QName(e).localname in RPR_ORDER else 99)
    kids = sorted(list(rp), key=key)
    for e in kids: rp.remove(e)
    rp.extend(kids)
    return rp


class Redliner:
    def __init__(self, track, author='Authors (Revision v2)', date='2026-10-21T00:00:00Z', start_id=900000):
        self.track, self.author, self.date, self._id = track, author, date, start_id

    # ------------------------------------------------------------ primitives
    def _attrs(self, el):
        self._id += 1
        el.set(w('id'), str(self._id)); el.set(w('author'), self.author); el.set(w('date'), self.date)
        return el

    def run(self, text, rPr=None, bold=None, italic=None, highlight=None):
        r = etree.Element(w('r'))
        rp = copy.deepcopy(rPr) if rPr is not None else etree.Element(w('rPr'))
        for tag, on in (('b', bold), ('i', italic)):
            if on is None: continue
            for e in rp.findall(w(tag)): rp.remove(e)
            if on: rp.insert(0, etree.Element(w(tag)))
        if highlight:
            for e in rp.findall(w('highlight')): rp.remove(e)
            hl = etree.SubElement(rp, w('highlight')); hl.set(w('val'), highlight)
        sort_rPr(rp)
        if len(rp): r.append(rp)
        t = etree.SubElement(r, w('t')); t.text = text; t.set(f'{{{XML}}}space', 'preserve')
        return r

    def ins(self, nodes):
        nodes = list(nodes)
        if not self.track: return nodes
        e = self._attrs(etree.Element(w('ins'))); e.extend(nodes); return [e]

    def dele(self, node):
        """Mark an existing w:r (or m:r) as deleted; returns replacement node list."""
        if not self.track: return []
        for t in node.iter(w('t')): t.tag = w('delText')
        for t in node.iter(w('instrText')): t.tag = w('delInstrText')
        e = self._attrs(etree.Element(w('del'))); e.append(node); return [e]

    @staticmethod
    def text(el): return ''.join(el.itertext())

    @staticmethod
    def first_rPr(p):
        for r in p.iter(w('r')):
            rp = r.find(w('rPr'))
            if r.find(w('t')) is not None:
                return copy.deepcopy(rp) if rp is not None else None
        return None

    def _segments_to_runs(self, segments, rPr):
        runs = []
        for s in segments:
            if isinstance(s, str): runs.append(self.run(s, rPr))
            else:  # (text, dict(bold=..., italic=..., highlight=...))
                runs.append(self.run(s[0], rPr, **s[1]))
        return runs

    # ------------------------------------------------------------ paragraph operations
    def _run_groups(self, p):
        """Consecutive direct text runs (ignoring proofErr / bookmarks between them)."""
        groups, cur = [], []
        for ch in p:
            tag = etree.QName(ch).localname
            if tag == 'r' and ch.find(w('t')) is not None and ch.find(w('drawing')) is None:
                cur.append(ch)
            elif tag in ('proofErr', 'bookmarkStart', 'bookmarkEnd', 'pPr'):
                continue
            else:
                if cur: groups.append(cur)
                cur = []
        if cur: groups.append(cur)
        return groups

    def replace_sub(self, p, old, new):
        """Tracked replacement of `old` by `new`; `old` may span consecutive text runs (not math)."""
        hits = []
        for g in self._run_groups(p):
            texts = [r.find(w('t')).text or '' for r in g]; full = ''.join(texts)
            if full.count(old) > 1: raise ValueError(f'replace_sub: {old[:60]!r} occurs more than once')
            if old in full: hits.append((g, texts, full.index(old)))
        if len(hits) != 1:
            raise ValueError(f'replace_sub: {len(hits)} hits for {old[:60]!r} in paragraph {self.text(p)[:60]!r}')
        g, texts, start = hits[0]; end = start + len(old); pos = 0; inserted = False
        for r, t in zip(g, texts):
            a, b = pos, pos + len(t); pos = b
            if b <= start or a >= end: continue
            rPr = r.find(w('rPr')); i, j = max(start, a) - a, min(end, b) - a
            out = []
            if t[:i]: out.append(self.run(t[:i], rPr))
            out += self.dele(self.run(t[i:j], rPr))
            if b >= end and not inserted:
                if new: out += self.ins([self.run(new, g[0].find(w('rPr')) if g[0].find(w('rPr')) is not None else rPr)])
                inserted = True
            if t[j:]: out.append(self.run(t[j:], rPr))
            parent = r.getparent(); idx = parent.index(r); parent.remove(r)
            for k, n in enumerate(out): parent.insert(idx + k, n)

    def replace_para(self, p, segments, rPr='first'):
        """Replace the full content of a text-only paragraph (no math, no drawings)."""
        if p.xpath('.//m:oMath|.//w:drawing', namespaces=NS):
            raise ValueError('replace_para on paragraph with math/drawing; use rewrite_para')
        rp = self.first_rPr(p) if rPr == 'first' else rPr
        for r in list(p.findall(w('r'))):
            idx = p.index(r); p.remove(r)
            for k, n in enumerate(self.dele(r)): p.insert(idx + k, n)
        p.extend(self.ins(self._segments_to_runs(segments, rp)))

    def delete_para(self, p):
        if not self.track:
            p.getparent().remove(p); return
        for r in list(p.xpath('./w:r|.//m:r', namespaces=NS)):
            parent = r.getparent(); idx = parent.index(r); parent.remove(r)
            for k, n in enumerate(self.dele(r)): parent.insert(idx + k, n)
        pPr = p.find(w('pPr'))
        if pPr is None: pPr = etree.Element(w('pPr')); p.insert(0, pPr)
        rp = pPr.find(w('rPr'))
        if rp is None: rp = etree.SubElement(pPr, w('rPr'))
        rp.insert(0, self._attrs(etree.Element(w('del'))))

    def new_para(self, like_p, segments, rPr='first'):
        p = etree.Element(w('p'))
        pPr = like_p.find(w('pPr'))
        if pPr is not None:
            pPr = copy.deepcopy(pPr)
            for e in pPr.findall(w('sectPr')): pPr.remove(e)
            for e in pPr.findall(w('rPr')): pPr.remove(e)
            p.append(pPr)
        if self.track:
            if pPr is None: pPr = etree.SubElement(p, w('pPr'))
            rp = etree.SubElement(pPr, w('rPr')); rp.append(self._attrs(etree.Element(w('ins'))))
        rp = self.first_rPr(like_p) if rPr == 'first' else rPr
        p.extend(self.ins(self._segments_to_runs(segments, rp)))
        return p

    def insert_after(self, anchor, like_p, segments, rPr='first'):
        p = self.new_para(like_p, segments, rPr); anchor.addnext(p); return p

    def rewrite_para(self, p, segments, rPr='first'):
        """Replace any paragraph: text-only -> in place; with math -> delete + insert plain-text paragraph."""
        if p.xpath('.//m:oMath|.//w:drawing', namespaces=NS):
            rp = self.first_rPr(p) if rPr == 'first' else rPr
            newp = self.new_para(p, segments, rp)
            p.addnext(newp); self.delete_para(p); return newp
        self.replace_para(p, segments, rPr); return p

    # ------------------------------------------------------------ tables
    def delete_table(self, tbl):
        if not self.track:
            tbl.getparent().remove(tbl); return
        for tr in tbl.findall(w('tr')):
            trPr = tr.find(w('trPr'))
            if trPr is None: trPr = etree.Element(w('trPr')); tr.insert(0 if tr.find(w('tblPrEx')) is None else 1, trPr)
            trPr.append(self._attrs(etree.Element(w('del'))))
            for r in list(tr.iter(w('r'))):
                parent = r.getparent(); idx = parent.index(r); parent.remove(r)
                for k, n in enumerate(self.dele(r)): parent.insert(idx + k, n)

    def make_table(self, like_tbl, rows, weights=None, bold_header=True, total_width=None, font_half_pt=18):
        """Build a new table styled like `like_tbl`. rows[0] is the header."""
        ncol = len(rows[0]); weights = weights or [1] * ncol
        tbl = etree.Element(w('tbl'))
        tblPr = copy.deepcopy(like_tbl.find(w('tblPr'))); tbl.append(tblPr)
        old_grid = like_tbl.find(w('tblGrid'))
        total = total_width or sum(int(g.get(w('w'))) for g in old_grid.findall(w('gridCol'))) or 9000
        for e in tblPr.findall(w('tblW')): tblPr.remove(e)
        tw = etree.Element(w('tblW')); tw.set(w('w'), str(total)); tw.set(w('type'), 'dxa')
        anchor_i = 1 if tblPr.find(w('tblStyle')) is not None else 0
        tblPr.insert(anchor_i, tw)
        widths = [int(total * x / sum(weights)) for x in weights]
        grid = etree.SubElement(tbl, w('tblGrid'))
        for wd in widths: etree.SubElement(grid, w('gridCol')).set(w('w'), str(wd))
        like_rows = like_tbl.findall(w('tr'))
        for ri, vals in enumerate(rows):
            src = like_rows[0] if ri == 0 else like_rows[min(1, len(like_rows) - 1)]
            src_tc = src.find(w('tc'))
            tr = etree.SubElement(tbl, w('tr'))
            trPr = copy.deepcopy(src.find(w('trPr'))) if src.find(w('trPr')) is not None else etree.Element(w('trPr'))
            for e in trPr.findall(w('ins')) + trPr.findall(w('del')): trPr.remove(e)
            if self.track: trPr.append(self._attrs(etree.Element(w('ins'))))
            if len(trPr): tr.append(trPr)
            src_p = src_tc.find(w('p')); src_rPr = self.first_rPr(src_tc)
            for ci, val in enumerate(vals):
                tc = etree.SubElement(tr, w('tc'))
                tcPr = copy.deepcopy(src_tc.find(w('tcPr'))) if src_tc.find(w('tcPr')) is not None else etree.Element(w('tcPr'))
                for e in tcPr.findall(w('gridSpan')) + tcPr.findall(w('vMerge')): tcPr.remove(e)
                tcW = tcPr.find(w('tcW'))
                if tcW is None: tcW = etree.Element(w('tcW')); tcPr.insert(0, tcW)
                tcW.set(w('w'), str(widths[ci])); tcW.set(w('type'), 'dxa')
                tc.append(tcPr)
                p = etree.SubElement(tc, w('p'))
                if src_p is not None and src_p.find(w('pPr')) is not None:
                    pPr = copy.deepcopy(src_p.find(w('pPr')))
                    for e in pPr.findall(w('rPr')): pPr.remove(e)
                    p.append(pPr)
                rr = self.run(str(val), src_rPr, bold=(bold_header if ri == 0 else None))
                rp = rr.find(w('rPr'))
                if rp is None: rp = etree.Element(w('rPr')); rr.insert(0, rp)
                for tag in ('sz', 'szCs'):
                    for e in rp.findall(w(tag)): rp.remove(e)
                    e = etree.SubElement(rp, w(tag)); e.set(w('val'), str(font_half_pt))
                sort_rPr(rp)
                p.extend(self.ins([rr]))
        return tbl
