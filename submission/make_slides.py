"""Build the presentation from the course template (Project_Presentation_Template.pptx).

The template's own slides are cloned so its fonts, title rule, bullets and slide numbers stay unchanged:
slide 1 (title layout) is filled in place; every content slide is a copy of the template's content slide
(title + slide number), with text, figures and code added in the body area.
"""
import copy
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from lxml import etree
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Emu, Inches, Pt
from PIL import Image

HERE = Path(__file__).parent
TEMPLATE = Path("D:/Downloads/Project_Presentation_Template.pptx")
OUT = HERE / "Orvessa_presentation.pptx"
FIG = HERE / "figures"
WEB = HERE / "web_images"
EQ = HERE / "equations"
EQ.mkdir(exist_ok=True)

REPO = "https://github.com/Lsidvd234/alien-rosetta"
NAVY = RGBColor(0x00, 0x20, 0x60)
INK2 = RGBColor(0x52, 0x51, 0x4E)
CODEBG = RGBColor(0xF4, 0xF5, 0xF7)
CODEBORDER = RGBColor(0xD9, 0xDC, 0xE1)

# ---------------------------------------------------------------- equations (rendered with matplotlib mathtext)
EQUATIONS = {
    "target": r"$z \;=\; m \,\Vert\,$ => $\,\Vert\, y, \qquad y = R(m)$",
    "loss": r"$\mathcal{L}(\theta) = -\sum_{(x,z)\in\mathcal{D}}\ \sum_{t=1}^{|z|} \log p_\theta(z_t \mid x, z_{<t}),"
            r"\qquad \theta = \{A, B, E_\Sigma\}$",
    "lora": r"$h = W x + B A x, \qquad B\in\mathbb{R}^{d\times r},\ A\in\mathbb{R}^{r\times d},\ r = 16$",
    "init": r"$E_\Sigma[g] \leftarrow e_{\mathrm{tok}(s(g))}$  (glyph starts at the embedding of its sound)",
    "errors": r"reading error: $\hat m \neq m$        writing error: $\hat m = m \ \wedge\ \hat y \neq y$",
    "coverage": r"coverage: $c_\mathcal{D}(k)$ = times key $k$ (word, structure, fact) occurs in training",
}


def render_equations():
    plt.rcParams["mathtext.fontset"] = "cm"
    for name, tex in EQUATIONS.items():
        fig = plt.figure(figsize=(0.01, 0.01))
        fig.text(0, 0, tex, fontsize=22, color="#0b0b0b")
        fig.savefig(EQ / f"{name}.png", dpi=220, bbox_inches="tight", pad_inches=0.04, transparent=True)
        plt.close(fig)


# ---------------------------------------------------------------- template helpers
def delete_slide(prs, slide):
    sld_ids = prs.slides._sldIdLst
    for sld_id in list(sld_ids):
        if prs.part.related_part(sld_id.rId) is slide.part:
            prs.part.drop_rel(sld_id.rId)
            sld_ids.remove(sld_id)
            return


def content_slide(prs, src, title):
    """A new slide on the template's content layout, with the template slide's title and number shapes."""
    s = prs.slides.add_slide(src.slide_layout)
    for ph in list(s.placeholders):
        ph._element.getparent().remove(ph._element)
    for shp in src.shapes:
        ph = shp.placeholder_format if shp.is_placeholder else None
        if ph is not None and ph.type is not None and int(ph.idx) in (0, 12):  # title, slide number
            s.shapes._spTree.append(copy.deepcopy(shp._element))
    title_shape = [x for x in s.shapes if x.is_placeholder and x.placeholder_format.idx == 0][0]
    r = title_shape.text_frame.paragraphs[0].runs[0]
    r.text = title
    return s


def _para_bullet(p, level=0, size=14, space_before=6, bullet=True):
    pPr = p._p.get_or_add_pPr()
    marL = 457200 + level * 342900
    pPr.set("marL", str(marL if bullet else 0))
    pPr.set("indent", str(-304800 if bullet else 0))
    for tag in ("a:lnSpc", "a:spcBef", "a:buClr", "a:buSzPts", "a:buChar", "a:buNone"):
        for el in pPr.findall(qn(tag)):
            pPr.remove(el)
    ln = etree.SubElement(pPr, qn("a:lnSpc"))
    etree.SubElement(ln, qn("a:spcPct")).set("val", "110000")
    sb = etree.SubElement(pPr, qn("a:spcBef"))
    etree.SubElement(sb, qn("a:spcPts")).set("val", str(space_before * 100))
    if bullet:
        bc = etree.SubElement(pPr, qn("a:buClr"))
        etree.SubElement(bc, qn("a:schemeClr")).set("val", "dk1")
        etree.SubElement(pPr, qn("a:buSzPts")).set("val", str(size * 100))
        etree.SubElement(pPr, qn("a:buChar")).set("char", "o" if level == 0 else "\u2013")
    else:
        etree.SubElement(pPr, qn("a:buNone"))


def text_box(slide, x, y, w, h, items, size=14, name="Body"):
    """items: list of str or (str, dict). A str starting with '## ' is a bold navy heading (no bullet).
    '**label** rest' makes the label bold. dict keys: level, bullet, size, color, italic."""
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tb.name = name
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = Inches(0.05)
    tf.margin_top = tf.margin_bottom = Inches(0.03)
    first = True
    for it in items:
        opts = {}
        if isinstance(it, tuple):
            it, opts = it
        p = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False
        sz = opts.get("size", size)
        heading = it.startswith("## ")
        if heading:
            it = it[3:]
            _para_bullet(p, bullet=False, size=sz, space_before=opts.get("before", 4))
            run = p.add_run()
            run.text = it
            run.font.size = Pt(opts.get("size", 20))
            run.font.bold = True
            run.font.color.rgb = NAVY
            continue
        _para_bullet(p, level=opts.get("level", 0), size=sz, bullet=opts.get("bullet", True),
                     space_before=opts.get("before", 6))
        parts = []
        if it.startswith("**") and "**" in it[2:]:
            lab, rest = it[2:].split("**", 1)
            parts = [(lab, True), (rest, False)]
        else:
            parts = [(it, False)]
        for txt, bold in parts:
            run = p.add_run()
            run.text = txt
            run.font.size = Pt(sz)
            run.font.bold = bold or opts.get("bold", False)
            run.font.italic = opts.get("italic", False)
            if "color" in opts:
                run.font.color.rgb = opts["color"]
    return tb


def picture(slide, path, x, y, w=None, h=None, name="Figure"):
    """Place an image inside the box (x, y, w, h) keeping its aspect ratio; centred in the box."""
    with Image.open(path) as im:
        iw, ih = im.size
    if w and h:
        scale = min(w / iw, h / ih)
        pw, ph = iw * scale, ih * scale
        px, py = x + (w - pw) / 2, y + (h - ph) / 2
    elif w:
        pw, ph, px, py = w, w * ih / iw, x, y
    else:
        pw, ph, px, py = h * iw / ih, h, x, y
    pic = slide.shapes.add_picture(str(path), Inches(px), Inches(py), Inches(pw), Inches(ph))
    pic.name = name
    return pic


def caption(slide, text, x, y, w, h=0.32, size=10):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tb.name = "Caption"
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = Inches(0.02)
    p = tf.paragraphs[0]
    run = p.add_run()
    run.text = text
    run.font.size = Pt(size)
    run.font.italic = True
    run.font.color.rgb = INK2
    return tb


def code_box(slide, code, x, y, w, h, size=10.5, label=None):
    box = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    box.name = "Code"
    box.adjustments[0] = 0.03
    box.fill.solid()
    box.fill.fore_color.rgb = CODEBG
    box.line.color.rgb = CODEBORDER
    box.line.width = Pt(0.75)
    box.shadow.inherit = False
    tf = box.text_frame
    tf.word_wrap = False
    tf.vertical_anchor = MSO_ANCHOR.TOP
    tf.margin_left = tf.margin_right = Inches(0.15)
    tf.margin_top = Inches(0.1)
    lines = code.strip("\n").split("\n")
    if label:
        lines = [label] + lines
    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = PP_ALIGN.LEFT
        run = p.add_run()
        run.text = line if line else " "
        run.font.name = "Courier New"
        run.font.size = Pt(size)
        is_label = label and i == 0
        is_comment = line.lstrip().startswith("#")
        run.font.bold = bool(is_label)
        run.font.color.rgb = NAVY if is_label else (RGBColor(0x2E, 0x7D, 0x32) if is_comment else RGBColor(0x1F, 0x1F, 0x1F))
    return box


def table(slide, rows, x, y, w, col_w, size=12, header=True, row_h=0.3):
    shp = slide.shapes.add_table(len(rows), len(rows[0]), Inches(x), Inches(y), Inches(w), Inches(row_h * len(rows)))
    shp.name = "Table"
    t = shp.table
    for j, cw in enumerate(col_w):
        t.columns[j].width = Inches(cw)
    for i, r in enumerate(rows):
        t.rows[i].height = Inches(row_h)
        for j, val in enumerate(r):
            c = t.cell(i, j)
            c.text = ""
            c.margin_left = c.margin_right = Inches(0.06)
            c.margin_top = c.margin_bottom = Inches(0.02)
            p = c.text_frame.paragraphs[0]
            run = p.add_run()
            run.text = str(val)
            run.font.size = Pt(size)
            run.font.bold = header and i == 0
            run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF) if (header and i == 0) else RGBColor(0x1F, 0x1F, 0x1F)
            c.fill.solid()
            c.fill.fore_color.rgb = NAVY if (header and i == 0) else (RGBColor(0xF4, 0xF5, 0xF7) if i % 2 else RGBColor(0xFF, 0xFF, 0xFF))
    return shp


# ---------------------------------------------------------------- code excerpts (from the project's code)
CODE_INIT = '''
def add_glyph_tokens(model, n_old, k, seed, init="mean", donor_ids=None):
    """Give k new glyph tokens a trainable embedding (rows n_old..n_old+k-1)."""
    emb = model.get_input_embeddings()
    w = emb.weight[:n_old].float()
    if init in ("latin", "latin_shuffled"):     # donor_ids: token ids of each glyph's sound
        g = torch.Generator(device="cpu").manual_seed(seed)
        rows = []
        for ids in donor_ids:
            if ids:                              # start at the embedding of the glyph's own sound
                rows.append(w[torch.tensor(ids, device=w.device)].mean(0))
            else:                                # no sound (virama): neutral start
                rows.append(w.mean(0) + torch.randn(w.shape[1], generator=g).to(w.device) * w.std(0) * 0.1)
        delta = torch.nn.Parameter(torch.stack(rows))
    ...
    model.model.embed_tokens = AddedEmbed(emb, n_old, delta)   # new rows trained, old rows frozen
    model.lm_head = AddedHead(model.lm_head, n_old, delta)     # tied output rows
    return delta
'''

CODE_LORA = '''
lcfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, task_type="CAUSAL_LM",
                  target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                  "gate_proj", "up_proj", "down_proj"])
delta = add_glyph_tokens(model, n_old, len(added), a.seed, a.new_tok_init, donors)
model = get_peft_model(model, lcfg)
delta.requires_grad_(True)                      # glyph embeddings train with the adapter
'''

CODE_SPLIT = '''
def split(gen, items):
    """items: (gold analysis, gold glyphs, model output 'analysis => glyphs')."""
    c = collections.Counter()
    for ana_g, gly_g, p in items:
        ana_p, _, gly_p = p.partition("=>")
        ana_p, gly_p = ana_p.strip(), gly_p.strip()
        a_ok, g_ok = ana_p == ana_g, gly_p == gly_g
        if a_ok:                                 # read right: any glyph error is a WRITING error
            c["both right" if g_ok else "translation error"] += 1
            continue
        if g_ok:
            c["lucky glyphs"] += 1
            continue
        own, missing = set(), False              # READING error: render the model's own (wrong) meaning
        try:
            cls, _ = from_analysis(ana_p, spell_name)
            for gs in itertools.product("mf", repeat=min(len(cls), 6)):
                own.add(" ".join(gen.rep.target(with_gender(cl, g)) for cl, g in zip(cls, gs + ("m",) * 9)))
        except KeyError:
            missing = True                       # a word key the language does not have
        c["understanding error" if gly_p in own else
          ("both wrong" if own else ("word not in language" if missing else "unreadable"))] += 1
    return c
'''

CODE_STEER = '''
# direction a -> b (e.g. tense PRES -> PAST): mean English-word state of b minus that of a, at layer L
d = stats[(feat, b)][L][0].mean(0) - stats[(feat, a)][L][0].mean(0)
vec = d / d.norm() * alpha * scale                 # scale = mean hidden-state norm of layer L

add = (mask * vec.view(1, 1, -1)).to(device)      # mask = the English-word positions of the prompt
def hook(_m, _inp, output):
    h = output[0] if isinstance(output, tuple) else output
    if h.shape[1] == n:                            # only while the prompt is read
        h = h + add.to(h.dtype)
        return (h,) + tuple(output[1:]) if isinstance(output, tuple) else h
    return output
handle = layers_of(model)[layer - 1].register_forward_hook(hook)
out = model.generate(input_ids=ids, attention_mask=att, do_sample=False, max_new_tokens=40)
handle.remove()
# flip = output analysis now has the target value; control = same-size random direction
'''


def build():
    render_equations()
    prs = Presentation(str(TEMPLATE))
    title_slide = prs.slides[0]
    src = prs.slides[1]                       # the template's content slide (title rule + number)
    originals = list(prs.slides)[1:]

    # ---------------- 1. title slide (filled in place)
    tp = [x for x in title_slide.shapes if x.is_placeholder and x.placeholder_format.idx == 0][0]
    p0 = tp.text_frame.paragraphs[0]
    p0.runs[0].text = "Orvessa"
    tp.top = tp.top + Inches(0.3)
    p1 = copy.deepcopy(p0._p)
    tp.text_frame._txBody.append(p1)
    para = tp.text_frame.paragraphs[1]
    para.runs[0].text = "How far does a small LLM learn a language it has never seen?"
    para.runs[0].font.size = Pt(24)
    sub = [x for x in title_slide.shapes if x.is_placeholder and x.placeholder_format.idx == 1][0]
    vals = ["Group Number: 14", "Names: Kavinraj", "Roll Numbers: 24IM10055"]
    for p, v in zip(sub.text_frame.paragraphs, vals):
        p.runs[0].text = v
    link = title_slide.shapes.add_textbox(Inches(3.2), Inches(6.45), Inches(6.9), Inches(0.4))
    link.name = "Code link"
    lp = link.text_frame.paragraphs[0]
    lp.alignment = PP_ALIGN.CENTER
    lr = lp.add_run()
    lr.text = "Code: github.com/Lsidvd234/alien-rosetta"
    lr.font.size = Pt(14)
    lr.font.color.rgb = NAVY
    lr.hyperlink.address = REPO
    title_slide.notes_slide.notes_text_frame.text = (
        "Orvessa: we invent a language, teach it to a small LLM with LoRA, and measure exactly where the "
        "learning stops.")

    # ---------------- 2. Problem statement: motivation and I/O
    s = content_slide(prs, src, "Problem Statement")
    text_box(s, 0.93, 1.3, 11.6, 5.2, [
        "## Why an invented language?",
        "Small LLMs are often adapted to new languages, scripts and notations with LoRA. Real languages make this "
        "hard to study: the model may have seen them, gold translations are noisy, and no single property can be "
        "changed on its own.",
        "We invent the language instead: an exact answer key, never seen in pretraining, and every property "
        "(grammar, script, vocabulary, structures in training) can be changed one at a time.",
        "## Research question",
        ("What does it take for a small LLM to learn a language it has never seen, and where does that learning "
         "stop?", {"italic": True}),
        "## Input \u2192 process \u2192 output",
        "**Input:** an English sentence x, e.g. \u201cTom is going to beat you.\u201d",
        "**Process:** Qwen3-0.6B + LoRA reads x, writes its meaning analysis m, then the translation y.",
        "**Output:** S=NAME.Tom V=beat.FUT O=PRON.2sg  =>  [glyphs: Tom you-ACC beat-FUT-3sg]",
    ], size=14)
    s.notes_slide.notes_text_frame.text = "Motivation, research question, and the input/process/output of the task."

    # ---------------- 3. Problem statement: mathematical formulation
    s = content_slide(prs, src, "Problem Statement")
    text_box(s, 0.93, 1.3, 11.6, 0.5, ["## Mathematical formulation"])
    rows = [
        ("A generator defines meanings m \u2208 M, an English renderer x = E(m, v) (v = surface variant) and a "
         "deterministic glyph renderer R (SOV order, endings, sound changes, 27 glyphs).", "target"),
        ("The model writes the meaning first, then the glyphs. Training minimises cross-entropy on the target only; "
         "only the LoRA matrices A, B and the new glyph embeddings E\u03a3 are trained.", "loss"),
        ("Because R is deterministic, every wrong output is exactly one of two kinds:", "errors"),
        ("The experiments vary how often each key occurs in training (0 = held out, a small dose, or many):",
         "coverage"),
    ]
    yy = 1.95
    for txt, eq in rows:
        text_box(s, 0.93, yy, 5.2, 1.0, [(txt, {"size": 13})])
        picture(s, EQ / f"{eq}.png", 6.35, yy + 0.05, w=6.3, h=0.85, name=f"Equation {eq}")
        yy += 1.13
    s.notes_slide.notes_text_frame.text = "The formal setup: generator, target string, loss, error split, coverage."

    # ---------------- 4. Problem statement: data pipeline
    s = content_slide(prs, src, "Problem Statement")
    text_box(s, 0.93, 1.3, 7.4, 5.3, [
        "## Data pipeline",
        "**1. Sample a meaning first** from families (statement, question, command, fragment; tense, aspect, "
        "modality, negation; subject/object kinds; 21 relation fields), so labels are right by construction.",
        "**2. Render English** with a surface variant (word order, voice, contractions, casing, typos, synonyms).",
        "**3. Render glyphs** deterministically: 11,078 roots + built words, case and agreement endings, sandhi, "
        "27 glyphs (Unicode Private Use Area), each one new token.",
        "**4. Word floor:** every word at least 12 times; 80,000 pairs + 2,500 refusals + 6,000 replay = 88,500.",
        "**5. One change per experiment:** hold structures out, put a dose back, spell words by a sound rule, "
        "replace 15% with real Tatoeba sentences, add everyday English, paraphrase.",
        "**Test sets:** 300 generated, 600 real Tatoeba sentences, and a sealed set of 147 + 25 free sentences "
        "written by the author, frozen by hash and used only as totals.",
    ], size=13)
    picture(s, WEB / "tamil_alphabet_chart.png", 8.75, 1.4, w=3.9, h=3.45, name="Tamil alphabet")
    caption(s, "The grammar and the abugida script are modelled on Tamil (subject\u2013object\u2013verb, case "
               "endings, one sign per syllable). Chart: public domain, Wikimedia Commons.", 8.65, 4.9, 4.1, 0.75)
    s.notes_slide.notes_text_frame.text = "How the training and test data are produced."

    # ---------------- 5. Architecture
    s = content_slide(prs, src, "Architecture")
    picture(s, WEB / "gpt_decoder_only.png", 0.95, 1.35, w=1.85, h=4.9, name="Decoder-only transformer")
    caption(s, "Decoder-only transformer. Marxav, CC BY-SA 4.0 (Wikimedia Commons).", 0.75, 6.28, 2.3, 0.5)
    picture(s, WEB / "lora_hu2021_fig1.png", 3.2, 1.45, w=3.3, h=2.9, name="LoRA")
    caption(s, "LoRA: only A and B are trained (Hu et al., 2021, Fig. 1).", 3.15, 4.38, 3.4, 0.35)
    picture(s, EQ / "lora.png", 3.2, 4.8, w=3.3, h=0.45, name="Equation LoRA")
    picture(s, EQ / "init.png", 3.0, 5.4, w=3.7, h=0.45, name="Equation init")
    text_box(s, 6.95, 1.3, 5.8, 5.4, [
        "## Model",
        "**Qwen3-0.6B-Base:** 28 layers, hidden size 1024, 16 query / 8 key-value heads, 596M parameters, frozen.",
        "**LoRA r = 16, \u03b1 = 32** on q, k, v, o, gate, up and down projections of every layer.",
        "**27 new glyph tokens** with trainable, tied input/output embeddings; each starts at the embedding of its "
        "sound (k, zh, aa, \u2026). 10.1M trainable parameters in total (1.7%).",
        "**Output format:** meaning analysis, then \u201c=>\u201d, then glyphs (analysis-first); direct models "
        "write glyphs only. A reader-only and a writer-only model split the two steps.",
        "## Training",
        "AdamW, lr 2e-4, no weight decay, 3% warm-up then linear decay, batch 32, 4 epochs (11,064 steps), fp16; "
        "about 6 GPU-hours per run on a free Kaggle T4. 165 runs in total.",
    ], size=13)
    s.notes_slide.notes_text_frame.text = "Base model, adapter, new tokens and training settings."

    # ---------------- 6-8. Contribution code
    s = content_slide(prs, src, "Contribution Code")
    text_box(s, 0.93, 1.3, 11.6, 0.9, [
        "## 1. New glyph tokens that start from their sound  (train/run.py)",
        ("Each invented glyph becomes one token whose embedding starts at the embedding of the Latin token for its "
         "sound; LoRA is then added on every projection.", {"bullet": False, "size": 13}),
    ])
    code_box(s, CODE_INIT, 0.93, 2.35, 11.5, 3.0, size=10)
    code_box(s, CODE_LORA, 0.93, 5.5, 11.5, 1.25, size=10)
    caption(s, "All code: github.com/Lsidvd234/alien-rosetta", 0.93, 6.82, 6.0, 0.3, size=10)
    s.notes_slide.notes_text_frame.text = "The glyph-token initialisation and the LoRA setup."

    s = content_slide(prs, src, "Contribution Code")
    text_box(s, 0.93, 1.3, 11.6, 0.9, [
        "## 2. Splitting every error into reading vs writing  (analysis/error_split.py)",
        ("Rendering a meaning is deterministic, so a wrong output is classified by re-rendering the model's own "
         "analysis with the language generator.", {"bullet": False, "size": 13}),
    ])
    code_box(s, CODE_SPLIT, 0.93, 2.35, 11.5, 4.4, size=10)
    s.notes_slide.notes_text_frame.text = "The error split used for claim D."

    s = content_slide(prs, src, "Contribution Code")
    text_box(s, 0.93, 1.3, 11.6, 0.9, [
        "## 3. Steering from the English hidden states  (analysis/inside_claim4_steer2.py)",
        ("A mean-difference direction is added to one layer while the model reads the English; the translation it "
         "then writes is checked for the target value.", {"bullet": False, "size": 13}),
    ])
    code_box(s, CODE_STEER, 0.93, 2.35, 11.5, 3.1, size=10)
    caption(s, "Shortened from the project code (loops over batches and settings removed).",
            0.93, 5.55, 11.0, 0.35, size=10)
    s.notes_slide.notes_text_frame.text = "The steering hook used for claim E."

    # ---------------- 9. Benchmark methods and evaluation
    s = content_slide(prs, src, "Benchmark Methods and Evaluation")
    text_box(s, 0.93, 1.3, 6.3, 0.5, ["## Methods compared (one change each)"])
    table(s, [
        ["Group", "Methods"],
        ["Baseline", "Qwen3-0.6B + LoRA, analysis first, 80k pairs, floor 12"],
        ["Training tricks", "focal loss (\u03b3 = 2), 6 epochs, word floor 25 (+ size-matched control)"],
        ["Data tricks", "paraphrased English (1.4%, 8%), two-stage rewriter"],
        ["Decoding", "beam search (4), majority vote of 5 samples"],
        ["Scale", "Qwen3-1.7B, same data"],
        ["English variety", "everyday English words; 15% real Tatoeba text"],
        ["Controls", "random / shuffled directions; bag-of-words probe; seeds"],
    ], 0.93, 1.9, 6.4, [1.65, 4.75], size=11, row_h=0.42)
    text_box(s, 7.65, 1.3, 5.1, 5.4, [
        "## Test sets",
        "**Generated 300:** same generator, never trained; shared by all runs.",
        "**Tatoeba 600:** real sentences; gold from an automatic reader (\u22480.6 right by hand-check).",
        "**Sealed 147 + 25:** free English by the author, frozen by hash, totals only.",
        "## Metrics and statistics",
        "Meaning exact, glyphs exact, word accuracy, refusal rate; reading vs writing errors.",
        "Paired McNemar tests, bootstrap 95% intervals, Holm correction over 21 tests.",
        "Predictions written before most runs; every number gives its run count.",
    ], size=13)
    s.notes_slide.notes_text_frame.text = "What we compare against and how we score."

    # ---------------- 10-16. Results and discussion
    R = "Results and Discussion"
    s = content_slide(prs, src, R)
    text_box(s, 0.93, 1.25, 11.6, 0.5, ["## A. Generalisation stops at what training covers  (strong)"])
    picture(s, FIG / "fig_heldout_structures.png", 0.93, 1.85, w=6.0, h=3.3, name="Held-out structures")
    picture(s, FIG / "fig_passive_dose.png", 7.15, 1.85, w=5.4, h=3.3, name="Passive dose")
    text_box(s, 0.93, 5.25, 11.6, 1.5, [
        "Held-out English structures score 0.00\u20130.09 (3 runs) while trained ones stay at 0.97\u20130.99.",
        "Putting back only 50 passives (0.06% of the data) lifts passive reading from 0.00 to 0.53 (0.66 on seed 1).",
        "Maths in glyph digits: removing one fact drops problems needing it to 0.05 (others 0.92); a scratchpad "
        "copies its longest training working (2 rows \u00d7 3 steps on 1,798 of 1,800 longer rows).",
    ], size=13)
    s.notes_slide.notes_text_frame.text = "Claim A."

    s = content_slide(prs, src, R)
    text_box(s, 0.93, 1.25, 11.6, 0.5, ["## B. A spelling rule transfers only when it covers the vocabulary  (moderate)"])
    picture(s, FIG / "fig_rule_vs_list.png", 0.93, 1.85, w=6.8, h=4.6, name="Rule vs list")
    text_box(s, 7.95, 1.85, 4.8, 4.8, [
        "Same letter-to-sound rules in both settings.",
        "**Minority (loanwords):** new loans 0.19; tripling the loans lifts seen ones to 0.82 but new ones only "
        "to 0.27 (the predicted +0.10 missed): a better list, not the rule.",
        "**Whole vocabulary:** 2,232 never-trained words spelled right 0.67 and recognised 0.87.",
        "One-letter mappings transfer better than multi-letter ones (0.23 vs 0.12, p = 0.005).",
        "1 run per condition; consistent with the Tolerance Principle; no threshold claimed.",
    ], size=13)
    s.notes_slide.notes_text_frame.text = "Claim B."

    s = content_slide(prs, src, R)
    text_box(s, 0.93, 1.25, 11.6, 0.5, ["## C. Only English variety helps real English  (strong on Tatoeba)"])
    picture(s, FIG / "fig_english_variety.png", 0.93, 1.8, w=8.0, h=3.65, name="English variety")
    text_box(s, 9.1, 1.85, 3.65, 4.9, [
        "15% real text: Tatoeba 0.62 \u2192 0.92 (2 seeds, p < 1e-48); sealed 0.47 \u2192 0.56 (paired p = 0.04 / "
        "0.01).",
        "Everyday words, no reader labels: +0.11 / +0.04.",
        "Focal loss, paraphrases, floor 25, 6 epochs, rewriter, beam, voting, 3\u00d7 model: null or worse.",
    ], size=13)
    text_box(s, 0.93, 5.55, 8.0, 1.2, [
        "Caveat: Tatoeba gold comes from the same automatic reader as the real training labels; a hand-check "
        "puts about +0.19 of the +0.31 on correct labels. The sealed gain is the fairer size.",
    ], size=12)
    s.notes_slide.notes_text_frame.text = "Claim C."

    s = content_slide(prs, src, R)
    text_box(s, 0.93, 1.25, 11.6, 0.5, ["## D. On real English, reading fails; with untrained words, both  (moderate)"])
    picture(s, FIG / "fig_reading_writing.png", 0.93, 1.8, w=8.3, h=3.6, name="Reading vs writing")
    text_box(s, 9.4, 1.85, 3.35, 4.9, [
        "Writing errors stay flat (0.24 / 0.31 / 0.26) while reading errors climb (0.21 / 0.38 / 0.52).",
        "Separate reader and writer: the reader causes 74% (Tatoeba) and 90% (sealed) of failures.",
        "The writer fails mostly where a word was never trained (0.20 vs 0.84); the sealed set agrees.",
    ], size=13)
    text_box(s, 0.93, 5.55, 8.3, 1.2, [
        "Mechanism for passives: a probe cannot find the doer in the hidden states of the model trained without "
        "passives (0.746 vs 0.704 surface baseline), but does in models trained on them (0.871, 0.950).",
    ], size=12)
    s.notes_slide.notes_text_frame.text = "Claim D."

    s = content_slide(prs, src, R)
    text_box(s, 0.93, 1.25, 11.6, 0.5, ["## D (continued). Inside the model: the errors are in the English representation"])
    picture(s, WEB / "actadd_turner2023_fig1.png", 0.93, 1.85, w=5.6, h=1.75, name="ActAdd concept")
    caption(s, "Activation addition, the method we apply (Turner et al., 2023, Fig. 1).", 0.93, 3.62, 5.6, 0.32)
    text_box(s, 6.85, 1.8, 5.9, 2.3, [
        "Probes read tense, negation, subject \u2026 from every model's hidden states (0.90\u20131.00).",
        "Where the output is wrong, the probe is wrong too (0.41 vs 0.79 majority), and 59% of the time it reads "
        "the same wrong value.",
        "Steering the English-side direction flips the output (tense 0.97\u20131.00 vs \u2264 0.02 random) and repairs "
        "11/12 and 10/10 real subject errors.",
    ], size=12)
    picture(s, FIG / "fig_steering.png", 1.9, 4.05, w=9.5, h=2.7, name="Steering results")
    s.notes_slide.notes_text_frame.text = (
        "Why reading fails: the translation is written from the English meaning representation, and most "
        "real-English errors are already wrong there.")

    s = content_slide(prs, src, R)
    text_box(s, 0.93, 1.25, 11.6, 0.5, ["## E. In maths, a missing fact is rebuilt only from facts seen in the same role"])
    picture(s, FIG / "fig_maths.png", 0.93, 1.85, w=7.6, h=4.6, name="Maths")
    text_box(s, 8.75, 1.85, 4.0, 4.9, [
        "Base-7 arithmetic in glyph digits; 5\u00d76 and 6\u00d75 removed from training.",
        "**1. Recall scratchpad:** every other step right, but 5\u00d76 steps only 0.49 (guessing; voting does "
        "not help).",
        "**2. Build from a smaller fact, one rule:** 0.08. It writes the block 5\u00d74 wrong, though it computes "
        "5\u00d74 right as a step.",
        "**3. Mix three building rules:** 0.77. Steps from blocks seen in that role: 99.6% right; never seen "
        "in that role: 17%.",
    ], size=13)
    s.notes_slide.notes_text_frame.text = "Claim E, part 1: how we got the model to build the missing fact."

    s = content_slide(prs, src, R)
    text_box(s, 0.93, 1.25, 11.6, 0.5, ["## E (continued). Five examples of the role are enough"])
    picture(s, FIG / "fig_maths_role.png", 0.93, 1.85, w=7.6, h=3.6, name="Maths role dose")
    text_box(s, 8.75, 1.85, 4.0, 4.6, [
        "**4. Role dose:** showing 5\u00d74 as a block in just 5 training workings makes 5\u00d76 right 0.91 "
        "(0.00 with none): a step, not a slope.",
        "The same recipe rebuilds 3\u00d74 (0.98) and 6\u00d76 (0.64); 15 of 18 forced routes follow the role rule.",
        "The dose never reaches the reverse order 6\u00d75 (\u2264 0.06), but with 5\u00d76 present, the "
        "never-seen 6\u00d75 is right 172/174.",
    ], size=13)
    text_box(s, 0.93, 5.7, 11.6, 0.9, [
        ("Caveat: one training run per condition, and same-seed maths runs vary by about \u00b10.1\u20130.2 (GPU "
         "non-determinism).", {"size": 12, "italic": True}),
    ])
    s.notes_slide.notes_text_frame.text = "Claim E, part 2: the role dose and the replicates."

    s = content_slide(prs, src, R)
    text_box(s, 0.93, 1.25, 11.6, 0.5, ["## Side study: does longer training help? No grokking"])
    picture(s, FIG / "fig_grokking.png", 0.93, 1.8, w=8.0, h=4.9, name="Grokking curves")
    text_box(s, 9.15, 1.85, 3.6, 4.9, [
        "Grokking = a late jump in test accuracy long after training saturates (Power et al., 2022).",
        "We trained 3 settings far past saturation with weight decay 1.0 (12k steps translation, 60k maths).",
        "No jump: test accuracy plateaus (0.73–0.84), longer problems stay at 0.09–0.11.",
        "The logit norm keeps rising: the model grows more confident in what it memorised instead of simplifying.",
    ], size=12)
    s.notes_slide.notes_text_frame.text = "Grokking side study: none observed."

    s = content_slide(prs, src, R)
    text_box(s, 0.93, 1.25, 5.75, 5.5, [
        "## Limitations",
        "Most extension results are single training runs; real text, steering and the 0/50 passive dose have two "
        "seeds.",
        "Tatoeba gold comes from an automatic reader (about a third wrong); the sealed set has 147 sentences "
        "(differences below \u22480.11 are noise).",
        "Same-seed maths runs differ by \u00b10.1\u20130.2 (non-deterministic attention backward); language runs "
        "reproduce exactly in deterministic mode.",
        "One model family and size, one invented grammar; claims B and D were partly formed after the first data.",
    ], size=13)
    text_box(s, 7.0, 1.25, 5.75, 5.5, [
        "## Future work",
        "Second training seed of the sound-rule language (claim B).",
        "Languages with 25% and 50% rule-spelled words: is the change from list to rule gradual or sharp?",
        "Three deterministic seeds of the role-rule maths runs and the 0 / 5 role doses (claim E), judged against "
        "the written thresholds.",
        "Confirm bit-identical maths runs with the attention fix, then re-judge the single-run maths thresholds.",
    ], size=13)
    s.notes_slide.notes_text_frame.text = "Limitations and future work."

    s = content_slide(prs, src, R)
    text_box(s, 0.93, 1.25, 11.6, 0.5, ["## Conclusion"])
    table(s, [
        ["Finding", "Evidence", "Strength"],
        ["A. Learning stops at training coverage", "held-out 0.00\u20130.09; 50 passives \u2192 0.53\u20130.66", "strong"],
        ["B. Rule only when it covers the vocabulary", "new words 0.67 vs new loans 0.19\u20130.27", "moderate"],
        ["C. Only English variety helps real English", "Tatoeba 0.62 \u2192 0.92; 8 alternatives null", "strong / moderate"],
        ["D. Reading fails; with new words, both", "reader 74\u201390% of failures; probes: errors already inside", "moderate"],
        ["E. Missing fact rebuilt only from its role", "mixed rules 0.77 vs one rule 0.08; 5 examples suffice", "moderate (1 run each)"],
    ], 0.93, 1.85, 11.6, [4.3, 5.1, 2.2], size=12, row_h=0.42)
    text_box(s, 0.93, 4.95, 11.6, 1.8, [
        "A small model learns the invented language itself well, but generalises only as far as its training "
        "covers: in structure, and in spelling only when the rule covers the vocabulary.",
        "On real English the limit is reading the English, and only more varied English in the data moves it, "
        "not loss functions, decoding, more epochs or a larger model.",
    ], size=14)
    s.notes_slide.notes_text_frame.text = "Conclusion."

    for sl in originals:
        delete_slide(prs, sl)
    prs.save(str(OUT))
    print("saved", OUT, "slides:", len(prs.slides))


if __name__ == "__main__":
    build()
