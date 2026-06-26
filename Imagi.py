import base64
import csv
import hashlib
import json
import math
import os
import shutil
import threading
import time
import webbrowser
import sys
from pathlib import Path
from tkinter import PhotoImage, filedialog, messagebox

import customtkinter as ctk
import numpy as np
from PIL import Image, ImageOps

import torch
from torchvision import models, transforms

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
APP_VERSION = "2.0"
APP_TITLE = "Imagi"
ASSET_ACORN_ICON = "imagi_acorn_icon.png"
ASSET_ACORN_ICO = "imagi_acorn_icon.ico"
ASSET_PROGRAM_NAME = "imagi_program_name.png"

COLOR_BG = "#2c1b14"
COLOR_SURFACE = "#4a2920"
COLOR_ACCENT = "#7b5c4a"
COLOR_ACCENT_HOVER = "#6d4f40"
COLOR_SOFT = "#d9cfc9"
COLOR_TEXT = "#e7ddd6"
COLOR_TEXT_DARK = "#2c1b14"
COLOR_MUTED = "#b9a79b"


def list_images(folder: Path):
    return sorted(
        [p for p in folder.rglob("*") if p.suffix.lower() in SUPPORTED_EXTENSIONS],
        key=lambda p: str(p).lower(),
    )


def file_signature(path: Path):
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "mtime_ns": stat.st_mtime_ns,
        "size": stat.st_size,
    }


def cache_key(paths, model_name, use_pretrained):
    payload = {
        "version": APP_VERSION,
        "model": model_name,
        "pretrained": bool(use_pretrained),
        "files": [file_signature(p) for p in paths],
    }
    text = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def get_device(use_gpu=False):
    if use_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_resnet18_feature_extractor(use_pretrained=True, use_gpu=False):
    device = get_device(use_gpu=use_gpu)

    if use_pretrained:
        try:
            weights = models.ResNet18_Weights.DEFAULT
            model = models.resnet18(weights=weights)
            preprocess = weights.transforms()
        except Exception as exc:
            raise RuntimeError(
                "Could not load pretrained ResNet18 weights. Check your internet connection "
                "or install the weights in advance."
            ) from exc
    else:
        model = models.resnet18(weights=None)
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    model.fc = torch.nn.Identity()
    model.eval()
    model.to(device)
    return model, preprocess, device


def get_clip_feature_extractor(use_gpu=False):
    """
    Optional CLIP backend. Install with: pip install open-clip-torch
    This keeps the default installation lightweight; ResNet remains the safe default.
    """
    try:
        import open_clip
    except Exception as exc:
        raise RuntimeError(
            "CLIP mode requires the optional package open-clip-torch. Install it with: pip install open-clip-torch"
        ) from exc

    device = get_device(use_gpu=use_gpu)
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="laion2b_s34b_b79k", device=device
    )
    model.eval()
    return model, preprocess, device


def load_and_preprocess(path: Path, preprocess):
    image = Image.open(path).convert("RGB")
    image = ImageOps.exif_transpose(image)
    return preprocess(image)


def extract_features(image_paths, progress_callback=None, model_name="resnet18", use_pretrained=True, use_gpu=False, batch_size=16):
    batch_size = max(1, int(batch_size))
    model_name = model_name.lower()

    if model_name == "clip":
        model, preprocess, device = get_clip_feature_extractor(use_gpu=use_gpu)
    else:
        model, preprocess, device = get_resnet18_feature_extractor(
            use_pretrained=use_pretrained,
            use_gpu=use_gpu,
        )

    feats = []
    valid_paths = []
    batch_tensors = []
    batch_paths = []
    processed = 0

    def flush_batch():
        nonlocal batch_tensors, batch_paths
        if not batch_tensors:
            return
        tensor = torch.stack(batch_tensors).to(device)
        with torch.no_grad():
            if model_name == "clip":
                out = model.encode_image(tensor)
                out = out / out.norm(dim=-1, keepdim=True)
            else:
                out = model(tensor)
        feats.extend(out.detach().cpu().numpy())
        valid_paths.extend(batch_paths)
        batch_tensors = []
        batch_paths = []

    for i, path in enumerate(image_paths):
        try:
            batch_tensors.append(load_and_preprocess(path, preprocess))
            batch_paths.append(path)
            if len(batch_tensors) >= batch_size:
                flush_batch()
        except Exception as exc:
            print(f"Skipping {path}: {exc}")

        processed += 1
        if progress_callback:
            progress_callback(processed, len(image_paths))

    flush_batch()

    if not feats:
        raise RuntimeError("No valid images found.")

    return np.vstack(feats), valid_paths


def load_or_extract_features(image_paths, cache_dir, use_cache=True, **kwargs):
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = cache_key(image_paths, kwargs.get("model_name", "resnet18"), kwargs.get("use_pretrained", True))
    npy_path = cache_dir / f"features_{key}.npy"
    meta_path = cache_dir / f"features_{key}.json"

    if use_cache and npy_path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        paths = [Path(p) for p in meta["valid_paths"]]
        features = np.load(npy_path)
        return features, paths, True

    features, valid_paths = extract_features(image_paths, **kwargs)
    np.save(npy_path, features)
    meta = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "app_version": APP_VERSION,
        "model_name": kwargs.get("model_name", "resnet18"),
        "valid_paths": [str(p) for p in valid_paths],
    }
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return features, valid_paths, False


def reduce_to_2d(features, spread=1.0, method="umap", n_neighbors=15, min_dist=0.08, metric="cosine", random_state=42):
    X = StandardScaler().fit_transform(features)
    method = method.lower()

    if method == "pca":
        coords = PCA(n_components=2, random_state=random_state).fit_transform(X)
    else:
        try:
            import umap
            reducer = umap.UMAP(
                n_components=2,
                n_neighbors=min(max(2, int(n_neighbors)), max(2, len(X) - 1)),
                min_dist=float(min_dist),
                metric=metric,
                random_state=int(random_state),
            )
            coords = reducer.fit_transform(X)
        except Exception as exc:
            print(f"UMAP unavailable or failed; using PCA instead: {exc}")
            coords = PCA(n_components=2, random_state=random_state).fit_transform(X)

    coords = np.asarray(coords, dtype=float)
    coords -= coords.min(axis=0)
    maxv = coords.max(axis=0)
    maxv[maxv == 0] = 1
    coords = coords / maxv
    coords = (coords - 0.5) * 1000 * float(spread)
    return coords


def cluster_features(features, min_cluster_size=5, method="hdbscan", dbscan_eps=2.5):
    X = StandardScaler().fit_transform(features)
    method = method.lower()

    if method == "none":
        return np.zeros(len(features), dtype=int)

    if method == "dbscan":
        labels = DBSCAN(
            eps=float(dbscan_eps),
            min_samples=max(2, int(min_cluster_size)),
        ).fit_predict(X)
        return labels.astype(int)

    try:
        import hdbscan
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=max(2, int(min_cluster_size)),
            min_samples=max(1, int(min_cluster_size) // 2),
            metric="euclidean",
        )
        labels = clusterer.fit_predict(X)
    except Exception as exc:
        print(f"HDBSCAN unavailable or failed; using DBSCAN instead: {exc}")
        labels = DBSCAN(
            eps=float(dbscan_eps),
            min_samples=max(2, int(min_cluster_size)),
        ).fit_predict(X)

    return labels.astype(int)


def make_thumbnail(path: Path, thumb_size=128, quality=85):
    img = Image.open(path).convert("RGB")
    img = ImageOps.exif_transpose(img)
    img.thumbnail((int(thumb_size), int(thumb_size)))
    return img


def image_to_data_uri(path: Path, thumb_size=128, quality=85):
    import io
    img = make_thumbnail(path, thumb_size=thumb_size, quality=quality)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=int(quality))
    data = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{data}"


def prepare_thumbnail_src(path: Path, out_dir: Path, thumb_size=128, quality=85, mode="embedded"):
    if mode == "embedded":
        return image_to_data_uri(path, thumb_size=thumb_size, quality=quality)

    thumb_dir = out_dir / "thumbnails"
    thumb_dir.mkdir(parents=True, exist_ok=True)
    safe_name = hashlib.sha1(str(path.resolve()).encode("utf-8")).hexdigest()[:12] + ".jpg"
    dest = thumb_dir / safe_name
    img = make_thumbnail(path, thumb_size=thumb_size, quality=quality)
    img.save(dest, format="JPEG", quality=int(quality))
    return "thumbnails/" + safe_name


def write_exports(items, out: Path):
    json_path = out.with_suffix(".data.json")
    csv_path = out.with_suffix(".data.csv")
    json_path.write_text(json.dumps(items, indent=2, ensure_ascii=False), encoding="utf-8")

    fields = ["name", "path", "x", "y", "cluster", "src"]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for item in items:
            writer.writerow({k: item.get(k, "") for k in fields})
    return json_path, csv_path


def build_html(items, html_title="Image map", point_size=64):
    data_json = json.dumps(items, ensure_ascii=False)
    template = r'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>__HTML_TITLE__</title>
<meta name="viewport" content="width=device-width, initial-scale=1" />
<style>
  :root { color-scheme: dark; }
  body { margin:0; font-family: Arial, Helvetica, sans-serif; background:#111; color:#eee; overflow:hidden; }
  #toolbar { position:fixed; top:12px; left:12px; right:336px; z-index:10; display:flex; flex-wrap:wrap; gap:8px; align-items:center; background:rgba(20,20,20,.92); border:1px solid rgba(255,255,255,.14); border-radius:12px; padding:10px; box-shadow:0 8px 32px rgba(0,0,0,.35); }
  button, select, input, textarea { background:#222; color:#eee; border:1px solid #555; border-radius:8px; padding:8px 10px; }
  button:hover { background:#333; cursor:pointer; }
  label { font-size:13px; opacity:.95; }
  #info { margin-left:auto; opacity:.75; font-size:13px; }
  #canvas { width:100vw; height:100vh; display:block; background:radial-gradient(circle at center,#1b1b1b 0%,#0d0d0d 100%); }
  .node image { cursor:grab; }
  .node image:active { cursor:grabbing; }
  .node text { font-size:10px; fill:#eee; paint-order:stroke; stroke:#000; stroke-width:3px; stroke-linejoin:round; pointer-events:none; }
  .hidden { display:none; }
  .selected rect { opacity:.95 !important; stroke:#fff; stroke-width:3px; }
  .near rect { opacity:.75 !important; stroke:#fff; stroke-width:2px; }
  .labels-hidden .node text { display:none; }
  #sidepanel { position:fixed; top:12px; right:12px; bottom:12px; width:292px; z-index:11; background:rgba(20,20,20,.96); border:1px solid rgba(255,255,255,.14); border-radius:12px; padding:12px; overflow:auto; box-shadow:0 8px 32px rgba(0,0,0,.35); }
  #preview { width:100%; max-height:240px; object-fit:contain; background:#000; border-radius:10px; margin:8px 0; }
  #meta { font-size:12px; line-height:1.45; opacity:.9; word-break:break-word; }
  #legend { max-height:160px; overflow:auto; font-size:12px; line-height:1.6; margin-top:10px; border-top:1px solid #444; padding-top:8px; }
  .legend-item { display:flex; align-items:center; gap:6px; cursor:pointer; }
  .swatch { width:12px; height:12px; border-radius:3px; display:inline-block; }
  textarea { width:100%; min-height:54px; box-sizing:border-box; margin-top:6px; }
  #selectedList { font-size:12px; opacity:.85; margin-top:8px; }
</style>
</head>
<body>
<div id="toolbar">
  <button onclick="resetView()">reset view</button>
  <button onclick="fitVisible()">fit visible</button>
  <button onclick="toggleLabels()">labels on/off</button>
  <button onclick="exportPNG()">export png</button>
  <button onclick="exportSVG()">export svg</button>
  <button onclick="downloadData('csv')">export csv</button>
  <button onclick="downloadData('json')">export json</button>
  <button onclick="downloadSelected()">export selected</button>
  <label>cluster <select id="clusterFilter" onchange="applyFilters()"></select></label>
  <label>search <input id="searchBox" type="text" placeholder="filename, tag, note..." oninput="applyFilters()" /></label>
  <label>size <input id="sizeSlider" type="range" min="24" max="180" value="__POINT_SIZE__" oninput="resizeNodes(this.value)" /></label>
  <label>opacity <input id="opacitySlider" type="range" min="10" max="100" value="100" oninput="setNodeOpacity(this.value)" /></label>
  <span id="info"></span>
</div>

<div id="sidepanel">
  <div style="font-size:18px; font-weight:700;">Image inspector</div>
  <img id="preview" alt="selected preview" />
  <div id="meta">Click an image to inspect it. Ctrl/cmd-click to multi-select.</div>
  <label>manual label/tag</label>
  <input id="manualLabel" type="text" placeholder="e.g. domestic scenes" oninput="saveCurrentAnnotation()" style="width:100%; box-sizing:border-box; margin:6px 0;" />
  <label>notes</label>
  <textarea id="notes" placeholder="qualitative notes..." oninput="saveCurrentAnnotation()"></textarea>
  <button onclick="highlightNearest()" style="width:100%; margin-top:8px;">highlight nearest</button>
  <button onclick="clearSelection()" style="width:100%; margin-top:8px;">clear selection</button>
  <div id="selectedList"></div>
  <div id="legend"></div>
</div>

<svg id="canvas" xmlns="http://www.w3.org/2000/svg"><g id="viewport"></g></svg>

<script>
const data = __DATA_JSON__;
const svg = document.getElementById("canvas");
const viewport = document.getElementById("viewport");
const clusterFilter = document.getElementById("clusterFilter");
const searchBox = document.getElementById("searchBox");
const info = document.getElementById("info");
const preview = document.getElementById("preview");
const meta = document.getElementById("meta");
const manualLabel = document.getElementById("manualLabel");
const notes = document.getElementById("notes");
const selectedList = document.getElementById("selectedList");
const legend = document.getElementById("legend");

let pointSize = Number("__POINT_SIZE__");
let transform = {x: window.innerWidth/2 - 140, y: window.innerHeight/2, k: 1};
let isPanning = false, panStart = null, dragNode = null, dragStart = null;
let labelsVisible = true;
let nodeOpacity = 1;
let currentIdx = null;
let selected = new Set();
let annotations = JSON.parse(localStorage.getItem(storageKey()) || "{}");

data.forEach((d, i) => { d.idx = i; d.manual_label = annotations[i]?.label || ""; d.notes = annotations[i]?.notes || ""; });

function storageKey() { return "image_map_annotations_" + document.title + "_" + data.length; }
function saveAnnotations() { localStorage.setItem(storageKey(), JSON.stringify(annotations)); }
function uniqueClusters() { return Array.from(new Set(data.map(d => d.cluster))).sort((a,b) => a-b); }
function clusterColor(c) { if (c === -1) return "#888888"; const hue=(c*57)%360; return `hsl(${hue},70%,55%)`; }
function updateTransform() { viewport.setAttribute("transform", `translate(${transform.x},${transform.y}) scale(${transform.k})`); }

function initClusterFilter() {
  clusterFilter.innerHTML = "";
  const all = document.createElement("option"); all.value="all"; all.textContent="all"; clusterFilter.appendChild(all);
  uniqueClusters().forEach(c => { const opt=document.createElement("option"); opt.value=String(c); opt.textContent=c===-1?"outliers (-1)":"cluster "+c; clusterFilter.appendChild(opt); });
}

function initLegend() {
  const counts = {};
  data.forEach(d => counts[d.cluster] = (counts[d.cluster] || 0) + 1);
  legend.innerHTML = "<div style='font-weight:700; margin-bottom:4px;'>Cluster legend</div>";
  uniqueClusters().forEach(c => {
    const div = document.createElement("div"); div.className="legend-item"; div.onclick=()=>{ clusterFilter.value=String(c); applyFilters(); fitVisible(); };
    div.innerHTML = `<span class="swatch" style="background:${clusterColor(c)}"></span><span>${c===-1?"outliers":"cluster "+c}: ${counts[c]} images</span>`;
    legend.appendChild(div);
  });
}

function render() {
  viewport.innerHTML = "";
  data.forEach((d, idx) => {
    const g = document.createElementNS("http://www.w3.org/2000/svg", "g");
    g.classList.add("node");
    g.setAttribute("data-cluster", d.cluster);
    g.setAttribute("data-name", (d.name + " " + (d.manual_label||"") + " " + (d.notes||"")).toLowerCase());
    g.setAttribute("transform", `translate(${d.x},${d.y})`);
    g.dataset.idx = idx;

    const rect = document.createElementNS("http://www.w3.org/2000/svg", "rect");
    rect.setAttribute("x", -pointSize/2 - 3); rect.setAttribute("y", -pointSize/2 - 3);
    rect.setAttribute("width", pointSize + 6); rect.setAttribute("height", pointSize + 6);
    rect.setAttribute("rx", 8); rect.setAttribute("fill", clusterColor(d.cluster)); rect.setAttribute("opacity", 0.35 * nodeOpacity);

    const img = document.createElementNS("http://www.w3.org/2000/svg", "image");
    img.setAttribute("href", d.src); img.setAttribute("x", -pointSize/2); img.setAttribute("y", -pointSize/2);
    img.setAttribute("width", pointSize); img.setAttribute("height", pointSize); img.setAttribute("opacity", nodeOpacity);
    img.setAttribute("preserveAspectRatio", "xMidYMid meet");

    const title = document.createElementNS("http://www.w3.org/2000/svg", "title");
    title.textContent = `${d.name} | cluster ${d.cluster}`; img.appendChild(title);

    const text = document.createElementNS("http://www.w3.org/2000/svg", "text");
    text.setAttribute("x", 0); text.setAttribute("y", pointSize/2 + 14); text.setAttribute("text-anchor", "middle");
    text.textContent = d.manual_label || (d.name.length > 24 ? d.name.slice(0,21)+"..." : d.name);

    g.appendChild(rect); g.appendChild(img); g.appendChild(text); viewport.appendChild(g);
  });
  document.body.classList.toggle("labels-hidden", !labelsVisible);
  applyFilters(); updateTransform();
}

function applyFilters() {
  const selectedCluster = clusterFilter.value;
  const q = searchBox.value.trim().toLowerCase();
  let visible = 0;
  document.querySelectorAll(".node").forEach(node => {
    const clusterOK = selectedCluster === "all" || node.getAttribute("data-cluster") === selectedCluster;
    const idx = Number(node.dataset.idx);
    const haystack = (data[idx].name + " " + data[idx].path + " " + (data[idx].manual_label||"") + " " + (data[idx].notes||"")).toLowerCase();
    const searchOK = !q || haystack.includes(q);
    const show = clusterOK && searchOK;
    node.classList.toggle("hidden", !show);
    if (show) visible += 1;
  });
  info.textContent = `${visible} / ${data.length} images | ${selected.size} selected`;
}

function resizeNodes(v) { pointSize = Number(v); render(); }
function setNodeOpacity(v) { nodeOpacity = Number(v)/100; render(); }
function toggleLabels() { labelsVisible = !labelsVisible; document.body.classList.toggle("labels-hidden", !labelsVisible); }
function resetView() { transform = {x: window.innerWidth/2 - 140, y: window.innerHeight/2, k: 1}; updateTransform(); }

function fitVisible() {
  const nodes = Array.from(document.querySelectorAll(".node:not(.hidden)")).map(n => data[Number(n.dataset.idx)]);
  if (!nodes.length) return;
  const xs = nodes.map(d => d.x), ys = nodes.map(d => d.y);
  const minX = Math.min(...xs), maxX = Math.max(...xs), minY = Math.min(...ys), maxY = Math.max(...ys);
  const w = Math.max(1, maxX-minX + pointSize*2), h = Math.max(1, maxY-minY + pointSize*2);
  const availableW = window.innerWidth - 340, availableH = window.innerHeight - 40;
  const k = Math.max(0.05, Math.min(20, 0.88 * Math.min(availableW/w, availableH/h)));
  transform.k = k;
  transform.x = (availableW/2) - ((minX+maxX)/2)*k;
  transform.y = (window.innerHeight/2) - ((minY+maxY)/2)*k;
  updateTransform();
}

function svgPoint(evt) { const pt=svg.createSVGPoint(); pt.x=evt.clientX; pt.y=evt.clientY; return pt.matrixTransform(viewport.getScreenCTM().inverse()); }

svg.addEventListener("wheel", evt => { evt.preventDefault(); const scale=evt.deltaY<0?1.1:.9; transform.k*=scale; transform.k=Math.max(.05,Math.min(20,transform.k)); updateTransform(); }, {passive:false});
svg.addEventListener("mousedown", evt => {
  const node = evt.target.closest && evt.target.closest(".node");
  if (node) {
    const idx = Number(node.dataset.idx);
    if (evt.ctrlKey || evt.metaKey) toggleSelected(idx); else inspect(idx, true);
    dragNode = node; const p = svgPoint(evt); dragStart = {idx, dx:p.x-data[idx].x, dy:p.y-data[idx].y};
  } else { isPanning=true; panStart={x:evt.clientX-transform.x, y:evt.clientY-transform.y}; }
});
svg.addEventListener("mousemove", evt => {
  if (dragNode && dragStart) { const p=svgPoint(evt); const d=data[dragStart.idx]; d.x=p.x-dragStart.dx; d.y=p.y-dragStart.dy; dragNode.setAttribute("transform", `translate(${d.x},${d.y})`); }
  else if (isPanning && panStart) { transform.x=evt.clientX-panStart.x; transform.y=evt.clientY-panStart.y; updateTransform(); }
});
window.addEventListener("mouseup", () => { dragNode=null; dragStart=null; isPanning=false; panStart=null; });

function inspect(idx, selectSingle=false) {
  currentIdx = idx;
  if (selectSingle) { selected = new Set([idx]); updateSelectedStyles(); }
  const d = data[idx]; preview.src = d.src;
  meta.innerHTML = `<b>${d.name}</b><br>cluster: ${d.cluster}<br>x/y: ${d.x.toFixed(2)}, ${d.y.toFixed(2)}<br>path: ${d.path}`;
  manualLabel.value = d.manual_label || ""; notes.value = d.notes || "";
  updateSelectedList(); applyFilters();
}
function toggleSelected(idx) { selected.has(idx) ? selected.delete(idx) : selected.add(idx); inspect(idx, false); updateSelectedStyles(); }
function updateSelectedStyles() { document.querySelectorAll(".node").forEach(n => n.classList.toggle("selected", selected.has(Number(n.dataset.idx)))); updateSelectedList(); applyFilters(); }
function clearSelection() { selected.clear(); currentIdx=null; preview.removeAttribute("src"); meta.textContent="Click an image to inspect it. Ctrl/cmd-click to multi-select."; manualLabel.value=""; notes.value=""; document.querySelectorAll(".node").forEach(n => n.classList.remove("selected", "near")); updateSelectedList(); applyFilters(); }
function updateSelectedList() { selectedList.textContent = selected.size ? `${selected.size} selected: ` + Array.from(selected).slice(0,6).map(i=>data[i].name).join(", ") + (selected.size>6?"...":"") : ""; }
function saveCurrentAnnotation() {
  if (currentIdx === null) return;
  annotations[currentIdx] = {label: manualLabel.value, notes: notes.value};
  data[currentIdx].manual_label = manualLabel.value; data[currentIdx].notes = notes.value;
  saveAnnotations();
  const node = document.querySelector(`.node[data-idx='${currentIdx}']`);
  if (node) node.setAttribute("data-name", (data[currentIdx].name + " " + data[currentIdx].manual_label + " " + data[currentIdx].notes).toLowerCase());
}
function highlightNearest() {
  if (currentIdx === null) return;
  document.querySelectorAll(".node").forEach(n => n.classList.remove("near"));
  const base = data[currentIdx];
  const nearest = data.map((d,i)=>({i, dist: Math.hypot(d.x-base.x, d.y-base.y)})).filter(o=>o.i!==currentIdx).sort((a,b)=>a.dist-b.dist).slice(0,12);
  nearest.forEach(o => { const n=document.querySelector(`.node[data-idx='${o.i}']`); if(n) n.classList.add("near"); });
}

function dataForDownload(onlySelected=false) {
  const arr = onlySelected ? Array.from(selected).map(i => data[i]) : data;
  return arr.map(d => ({name:d.name, path:d.path, x:d.x, y:d.y, cluster:d.cluster, manual_label:d.manual_label||"", notes:d.notes||""}));
}
function downloadBlob(filename, text, type) { const blob=new Blob([text], {type}); const url=URL.createObjectURL(blob); const a=document.createElement("a"); a.href=url; a.download=filename; document.body.appendChild(a); a.click(); document.body.removeChild(a); URL.revokeObjectURL(url); }
function downloadData(format) {
  const rows = dataForDownload(false);
  if (format === "json") return downloadBlob("image_map_data.json", JSON.stringify(rows, null, 2), "application/json");
  const header = ["name","path","x","y","cluster","manual_label","notes"];
  const csv = [header.join(",")].concat(rows.map(r => header.map(h => '"' + String(r[h] ?? "").replaceAll('"','""') + '"').join(","))).join("\n");
  downloadBlob("image_map_data.csv", csv, "text/csv;charset=utf-8");
}
function downloadSelected() { const rows=dataForDownload(true); downloadBlob("image_map_selected.json", JSON.stringify(rows, null, 2), "application/json"); }

function visibleExportNodes() {
  return Array.from(document.querySelectorAll(".node:not(.hidden)"));
}

function buildExportClone() {
  const clone = svg.cloneNode(true);
  clone.setAttribute("xmlns", "http://www.w3.org/2000/svg");
  clone.setAttribute("xmlns:xlink", "http://www.w3.org/1999/xlink");
  clone.setAttribute("width", window.innerWidth);
  clone.setAttribute("height", window.innerHeight);
  clone.setAttribute("viewBox", `0 0 ${window.innerWidth} ${window.innerHeight}`);

  clone.querySelectorAll(".hidden").forEach(n => n.remove());

  const bg = document.createElementNS("http://www.w3.org/2000/svg", "rect");
  bg.setAttribute("x", "0");
  bg.setAttribute("y", "0");
  bg.setAttribute("width", String(window.innerWidth));
  bg.setAttribute("height", String(window.innerHeight));
  bg.setAttribute("fill", "#111111");
  clone.insertBefore(bg, clone.firstChild);

  if (!labelsVisible) {
    clone.querySelectorAll("text").forEach(t => t.remove());
  }

  const clonedNodes = Array.from(clone.querySelectorAll(".node"));
  clonedNodes.forEach(node => {
    const idx = Number(node.dataset.idx);
    const img = node.querySelector("image");
    if (!img || Number.isNaN(idx) || !data[idx]) return;

    const exportSrc = data[idx].export_src || data[idx].src;
    img.setAttribute("href", exportSrc);
    img.setAttributeNS("http://www.w3.org/1999/xlink", "href", exportSrc);

    const textEl = node.querySelector("text");
    if (textEl) {
      textEl.setAttribute("fill", "#eeeeee");
      textEl.setAttribute("font-family", "Arial, Helvetica, sans-serif");
      textEl.setAttribute("paint-order", "stroke");
      textEl.setAttribute("stroke", "#000000");
      textEl.setAttribute("stroke-width", "3");
      textEl.setAttribute("stroke-linejoin", "round");
    }
  });

  return clone;
}

function exportSVG() {
  const clone = buildExportClone();
  const source = new XMLSerializer().serializeToString(clone);
  downloadBlob("image_map.svg", source, "image/svg+xml;charset=utf-8");
}

function exportPNG() {
  const clone = buildExportClone();
  const source = new XMLSerializer().serializeToString(clone);
  const url = URL.createObjectURL(new Blob([source], {type:"image/svg+xml;charset=utf-8"}));

  const image = new Image();
  image.onload = function() {
    const canvas = document.createElement("canvas");
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
    const ctx = canvas.getContext("2d");
    ctx.fillStyle = "#111111";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(image, 0, 0);
    URL.revokeObjectURL(url);

    canvas.toBlob(blob => {
      const pngUrl = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = pngUrl;
      a.download = "image_map.png";
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(pngUrl);
    }, "image/png");
  };
  image.src = url;
}

initClusterFilter(); initLegend(); render(); fitVisible();
</script>
</body>
</html>'''
    return (
        template
        .replace("__HTML_TITLE__", html_title)
        .replace("__POINT_SIZE__", str(int(point_size)))
        .replace("__DATA_JSON__", data_json)
    )


class ImageMapApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title(APP_TITLE)
        self.geometry("900x780")
        self.resizable(True, True)

        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.assets_dir = Path(__file__).resolve().parent
        self._icon_image = None
        self._brand_image = None
        self.configure(fg_color=COLOR_BG)
        self._setup_branding()

        self.folder_path = ctk.StringVar(value="")
        self.output_path = ctk.StringVar(value=str(Path.cwd() / "output" / "image_map.html"))
        self.preset = ctk.StringVar(value="Balanced")
        self.model_name = ctk.StringVar(value="resnet18")
        self.layout_method = ctk.StringVar(value="umap")
        self.cluster_method = ctk.StringVar(value="hdbscan")
        self.thumbnail_mode = ctk.StringVar(value="external")
        self.spread = ctk.DoubleVar(value=1.0)
        self.point_size = ctk.IntVar(value=72)
        self.min_cluster_size = ctk.IntVar(value=5)
        self.thumb_size = ctk.IntVar(value=128)
        self.thumb_quality = ctk.IntVar(value=85)
        self.batch_size = ctk.IntVar(value=16)
        self.umap_neighbors = ctk.IntVar(value=15)
        self.umap_min_dist = ctk.DoubleVar(value=0.08)
        self.dbscan_eps = ctk.DoubleVar(value=2.5)
        self.random_state = ctk.IntVar(value=42)
        self.use_pretrained = ctk.BooleanVar(value=True)
        self.use_gpu = ctk.BooleanVar(value=False)
        self.use_cache = ctk.BooleanVar(value=True)
        self.open_when_done = ctk.BooleanVar(value=True)

        self._build_ui()

    def _setup_branding(self):
        if sys.platform.startswith("win"):
            try:
                import ctypes
                ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("imagi.desktop.app")
            except Exception:
                pass

        icon_path = self.assets_dir / ASSET_ACORN_ICON
        ico_path = self.assets_dir / ASSET_ACORN_ICO
        if icon_path.exists():
            try:
                self._icon_image = PhotoImage(file=str(icon_path))
                self.iconphoto(True, self._icon_image)
            except Exception:
                pass
        if sys.platform.startswith("win") and ico_path.exists():
            try:
                self.iconbitmap(default=str(ico_path))
            except Exception:
                pass

    def _load_brand_image(self, width=520):
        brand_path = self.assets_dir / ASSET_PROGRAM_NAME
        if not brand_path.exists():
            return None
        try:
            image = Image.open(brand_path).convert("RGBA")
            ratio = width / max(image.width, 1)
            size = (width, max(1, int(image.height * ratio)))
            self._brand_image = ctk.CTkImage(light_image=image, dark_image=image, size=size)
            return self._brand_image
        except Exception:
            return None

    def _build_ui(self):
        scroll = ctk.CTkScrollableFrame(self, corner_radius=16, fg_color=COLOR_BG)
        scroll.pack(fill="both", expand=True, padx=18, pady=18)

        brand_row = ctk.CTkFrame(scroll, fg_color="transparent")
        brand_row.pack(anchor="w", padx=18, pady=(18, 8))

        brand_image = self._load_brand_image(width=520)
        if brand_image is not None:
            ctk.CTkLabel(brand_row, text="", image=brand_image).pack(side="left")
        else:
            ctk.CTkLabel(
                brand_row,
                text=APP_TITLE,
                font=ctk.CTkFont(size=32, weight="bold"),
                text_color=COLOR_ACCENT,
            ).pack(side="left")

        subtitle = ctk.CTkLabel(
            scroll,
            text="Interactive visual maps with cache, batch extraction, annotations, CSV/JSON export, and richer HTML inspection.",
            text_color=COLOR_MUTED,
        )
        subtitle.pack(anchor="w", padx=18, pady=(0, 18))

        self._path_row(scroll, "Image folder", self.folder_path, self.select_folder, "Select folder")
        self._path_row(scroll, "Output HTML path", self.output_path, self.select_output, "Save as...")

        presets = ctk.CTkFrame(scroll, corner_radius=12, fg_color=COLOR_SURFACE)
        presets.pack(fill="x", padx=18, pady=(12, 6))
        ctk.CTkLabel(presets, text="Preset", width=120, anchor="w", text_color=COLOR_TEXT).pack(side="left", padx=14, pady=10)
        ctk.CTkOptionMenu(
            presets,
            values=["Fast / large dataset", "Balanced", "Presentation export", "Experimental / semantic"],
            variable=self.preset,
            command=self.apply_preset,
            fg_color=COLOR_ACCENT,
            button_color=COLOR_SURFACE,
            button_hover_color=COLOR_ACCENT_HOVER,
            text_color=COLOR_TEXT,
            dropdown_fg_color=COLOR_SURFACE,
            dropdown_hover_color=COLOR_ACCENT_HOVER,
            dropdown_text_color=COLOR_TEXT,
        ).pack(side="left", padx=8)

        params = ctk.CTkFrame(scroll, corner_radius=12, fg_color=COLOR_SURFACE)
        params.pack(fill="x", padx=18, pady=12)
        self._slider(params, "Distance / spread", self.spread, 0.4, 4.0)
        self._slider(params, "Point / image size", self.point_size, 24, 180)
        self._slider(params, "Minimum cluster size", self.min_cluster_size, 2, 60)
        self._slider(params, "Thumbnail size", self.thumb_size, 64, 384)
        self._slider(params, "Thumbnail quality", self.thumb_quality, 40, 95)
        self._slider(params, "Batch size", self.batch_size, 1, 64)
        self._slider(params, "UMAP neighbors", self.umap_neighbors, 2, 80)
        self._slider(params, "UMAP min_dist", self.umap_min_dist, 0.0, 0.8)
        self._slider(params, "DBSCAN eps", self.dbscan_eps, 0.2, 8.0)
        self._slider(params, "Random seed", self.random_state, 1, 999)

        options = ctk.CTkFrame(scroll, corner_radius=12, fg_color=COLOR_SURFACE)
        options.pack(fill="x", padx=18, pady=12)
        self._option_row(options, "Feature model", self.model_name, ["resnet18", "clip"])
        self._option_row(options, "Layout", self.layout_method, ["umap", "pca"])
        self._option_row(options, "Clustering", self.cluster_method, ["hdbscan", "dbscan", "none"])
        self._option_row(options, "Thumbnails", self.thumbnail_mode, ["external", "embedded"])

        checks = ctk.CTkFrame(scroll, fg_color="transparent")
        checks.pack(fill="x", padx=18, pady=4)
        for text, var in [
            ("Use pretrained ResNet18 weights", self.use_pretrained),
            ("Use GPU if available", self.use_gpu),
            ("Use feature cache", self.use_cache),
            ("Open map when done", self.open_when_done),
        ]:
            ctk.CTkCheckBox(
                checks,
                text=text,
                variable=var,
                text_color=COLOR_TEXT,
                fg_color=COLOR_ACCENT,
                hover_color=COLOR_ACCENT_HOVER,
                border_color=COLOR_ACCENT,
                checkmark_color=COLOR_TEXT,
            ).pack(anchor="w", pady=3)

        self.progress = ctk.CTkProgressBar(scroll, progress_color=COLOR_ACCENT, fg_color=COLOR_SURFACE)
        self.progress.set(0)
        self.progress.pack(fill="x", padx=18, pady=(20, 8))
        self.status = ctk.CTkLabel(scroll, text="Ready.", text_color=COLOR_MUTED, justify="left")
        self.status.pack(anchor="w", padx=18, pady=(0, 12))

        buttons = ctk.CTkFrame(scroll, fg_color="transparent")
        buttons.pack(fill="x", padx=18, pady=(8, 18))
        self.generate_button = ctk.CTkButton(
            buttons,
            text="Generate map",
            height=42,
            command=self.run_generation_thread,
            fg_color=COLOR_ACCENT,
            hover_color=COLOR_ACCENT_HOVER,
            text_color=COLOR_TEXT,
        )
        self.generate_button.pack(side="left", padx=(0, 8))
        ctk.CTkButton(
            buttons,
            text="Open output folder",
            height=42,
            command=self.open_output_folder,
            fg_color=COLOR_SURFACE,
            hover_color=COLOR_ACCENT_HOVER,
            text_color=COLOR_TEXT,
        ).pack(side="left")

    def _path_row(self, parent, placeholder, variable, command, button_text):
        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.pack(fill="x", padx=18, pady=6)
        ctk.CTkEntry(
            row,
            textvariable=variable,
            placeholder_text=placeholder,
            fg_color=COLOR_SURFACE,
            text_color=COLOR_TEXT,
            placeholder_text_color=COLOR_MUTED,
            border_color=COLOR_ACCENT,
        ).pack(side="left", fill="x", expand=True, padx=(0, 8))
        ctk.CTkButton(
            row,
            text=button_text,
            command=command,
            fg_color=COLOR_ACCENT,
            hover_color=COLOR_ACCENT_HOVER,
            text_color=COLOR_TEXT,
        ).pack(side="right")

    def _option_row(self, parent, label, variable, values):
        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.pack(fill="x", padx=14, pady=8)
        ctk.CTkLabel(row, text=label, width=160, anchor="w", text_color=COLOR_TEXT).pack(side="left")
        ctk.CTkOptionMenu(
            row,
            values=values,
            variable=variable,
            fg_color=COLOR_ACCENT,
            button_color=COLOR_SURFACE,
            button_hover_color=COLOR_ACCENT_HOVER,
            text_color=COLOR_TEXT,
            dropdown_fg_color=COLOR_SURFACE,
            dropdown_hover_color=COLOR_ACCENT_HOVER,
            dropdown_text_color=COLOR_TEXT,
        ).pack(side="left", padx=12)

    def _slider(self, parent, label, variable, minv, maxv):
        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.pack(fill="x", padx=14, pady=8)
        ctk.CTkLabel(row, text=label, width=160, anchor="w", text_color=COLOR_TEXT).pack(side="left")
        value_label = ctk.CTkLabel(row, text=str(variable.get()), width=60, text_color=COLOR_MUTED)
        value_label.pack(side="right")

        def update_value(v):
            if isinstance(variable, ctk.IntVar):
                variable.set(int(float(v)))
            else:
                variable.set(round(float(v), 2))
            value_label.configure(text=str(variable.get()))

        slider = ctk.CTkSlider(
            row,
            from_=minv,
            to=maxv,
            command=update_value,
            fg_color=COLOR_SURFACE,
            progress_color=COLOR_ACCENT,
            button_color=COLOR_ACCENT,
            button_hover_color=COLOR_SURFACE,
        )
        slider.set(variable.get())
        slider.pack(side="left", fill="x", expand=True, padx=12)

    def apply_preset(self, value):
        if value == "Fast / large dataset":
            self.layout_method.set("pca"); self.thumbnail_mode.set("external"); self.thumb_size.set(96); self.batch_size.set(32); self.point_size.set(56)
        elif value == "Presentation export":
            self.layout_method.set("umap"); self.thumbnail_mode.set("embedded"); self.thumb_size.set(192); self.point_size.set(96); self.spread.set(1.4)
        elif value == "Experimental / semantic":
            self.model_name.set("clip"); self.layout_method.set("umap"); self.thumbnail_mode.set("external"); self.umap_neighbors.set(30)
        else:
            self.model_name.set("resnet18"); self.layout_method.set("umap"); self.thumbnail_mode.set("external"); self.thumb_size.set(128); self.batch_size.set(16)
        messagebox.showinfo("Preset applied", "Preset values updated. Some slider labels refresh after you move them or restart the app.")

    def select_folder(self):
        folder = filedialog.askdirectory(title="Select image folder")
        if folder:
            self.folder_path.set(folder)

    def select_output(self):
        path = filedialog.asksaveasfilename(title="Save HTML as", defaultextension=".html", filetypes=[("HTML files", "*.html")])
        if path:
            self.output_path.set(path)

    def set_status(self, text):
        self.status.configure(text=text)
        self.update_idletasks()

    def set_progress(self, value):
        self.progress.set(value)
        self.update_idletasks()

    def run_generation_thread(self):
        threading.Thread(target=self.generate_map, daemon=True).start()

    def generate_map(self):
        started = time.time()
        try:
            self.generate_button.configure(state="disabled")
            self.set_progress(0)

            folder = Path(self.folder_path.get())
            if not folder.exists() or not folder.is_dir():
                messagebox.showerror("Missing folder", "Please select a valid image folder.")
                return

            out = Path(self.output_path.get())
            out.parent.mkdir(parents=True, exist_ok=True)
            image_paths = list_images(folder)
            if not image_paths:
                messagebox.showerror("No images", "No supported images found in this folder.")
                return

            cache_dir = out.parent / ".image_map_cache"
            self.set_status(f"Found {len(image_paths)} images. Extracting or loading features...")

            def progress_cb(done, total):
                self.set_status(f"Extracting features: {done}/{total}")
                self.set_progress(done / max(total, 1) * 0.45)

            features, valid_paths, loaded_cache = load_or_extract_features(
                image_paths,
                cache_dir=cache_dir,
                use_cache=self.use_cache.get(),
                progress_callback=progress_cb,
                model_name=self.model_name.get(),
                use_pretrained=self.use_pretrained.get(),
                use_gpu=self.use_gpu.get(),
                batch_size=self.batch_size.get(),
            )
            if loaded_cache:
                self.set_status("Loaded features from cache.")
                self.set_progress(0.45)

            self.set_status("Reducing dimensions...")
            self.set_progress(0.58)
            coords = reduce_to_2d(
                features,
                spread=self.spread.get(),
                method=self.layout_method.get(),
                n_neighbors=self.umap_neighbors.get(),
                min_dist=self.umap_min_dist.get(),
                random_state=self.random_state.get(),
            )

            self.set_status("Clustering...")
            self.set_progress(0.68)
            labels = cluster_features(
                features,
                min_cluster_size=self.min_cluster_size.get(),
                method=self.cluster_method.get(),
                dbscan_eps=self.dbscan_eps.get(),
            )

            if self.thumbnail_mode.get() == "external":
                thumb_dir = out.parent / "thumbnails"
                if thumb_dir.exists():
                    shutil.rmtree(thumb_dir)

            self.set_status("Preparing thumbnails and data exports...")
            items = []
            for i, (path, xy, label) in enumerate(zip(valid_paths, coords, labels)):
                thumbnail_src = prepare_thumbnail_src(
                    path,
                    out_dir=out.parent,
                    thumb_size=self.thumb_size.get(),
                    quality=self.thumb_quality.get(),
                    mode=self.thumbnail_mode.get(),
                )
                export_src = thumbnail_src if str(thumbnail_src).startswith("data:") else image_to_data_uri(
                    path,
                    thumb_size=self.thumb_size.get(),
                    quality=self.thumb_quality.get(),
                )
                items.append({
                    "name": path.name,
                    "path": str(path),
                    "x": float(xy[0]),
                    "y": float(xy[1]),
                    "cluster": int(label),
                    "src": thumbnail_src,
                    "export_src": export_src,
                })
                self.set_progress(0.68 + ((i + 1) / len(valid_paths)) * 0.22)

            json_path, csv_path = write_exports(items, out)
            html = build_html(items, html_title="Image map", point_size=self.point_size.get())
            out.write_text(html, encoding="utf-8")

            clusters = len(set(labels)) - (1 if -1 in set(labels) else 0)
            outliers = int(np.sum(labels == -1))
            elapsed = round(time.time() - started, 1)
            self.set_progress(1.0)
            self.set_status(
                f"Done in {elapsed}s. Processed {len(valid_paths)}/{len(image_paths)} images. "
                f"Clusters: {clusters}; outliers: {outliers}.\nSaved: {out}\nData: {csv_path.name}, {json_path.name}"
            )
            if self.open_when_done.get():
                webbrowser.open(out.resolve().as_uri())

        except Exception as exc:
            messagebox.showerror("Error", str(exc))
            self.set_status(f"Error: {exc}")
        finally:
            self.generate_button.configure(state="normal")

    def open_output_folder(self):
        out = Path(self.output_path.get())
        folder = out.parent if out.suffix else out
        folder.mkdir(parents=True, exist_ok=True)
        webbrowser.open(folder.resolve().as_uri())


if __name__ == "__main__":
    app = ImageMapApp()
    app.mainloop()
