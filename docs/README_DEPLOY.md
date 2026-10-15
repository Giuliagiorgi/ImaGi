# ImaGi Browser Edition — GitHub Pages

## What is included

This is a separate, zero-hosting-cost browser edition. It does not replace the original `Imagi.py` desktop application. It uses Transformers.js to run CLIP image embeddings on the user's machine, computes PCA or UMAP in JavaScript, DBSCAN or no clustering, then launches the original ImaGi interactive HTML map viewer. Images are processed locally; only CLIP model code/weights are downloaded from Hugging Face/CDNs.

## Deploy to Giuliagiorgi/ImaGi

1. Create a new `docs/` folder at the root of the repository.
2. Copy **all files** in this zip into that `docs/` folder: `index.html`, `app.js`, `viewer_template.html`, `imagi_acorn_icon.png`, `imagi_program_name.png`. (The README file is optional.)
3. Commit and push to `main`.
4. In the GitHub repository open **Settings → Pages**. Set **Build and deployment → Deploy from a branch**, choose **main**, folder **/docs**, Save.
5. The expected page address is `https://giuliagiorgi.github.io/ImaGi/`. Wait for the Pages deployment and test with 3–10 photos first.

No Hugging Face Spaces, paid hosting, servers, or student installation required. GitHub Pages serves static files. The model is fetched by each student's browser.

## Important distinctions and limitations

- Current inference: CLIP `Xenova/clip-vit-base-patch32`, not original ResNet18. HDBSCAN has **not** been ported; DBSCAN is offered instead. Model embeddings and numerical results may differ from the desktop edition. In-browser UMAP may not match Python UMAP exactly.
- No imposed maximum number of images. Memory, model download size, and per-image processing time depend on students' computers. The basic PCA algorithm in this prototype uses O(N²) memory; DBSCAN also uses O(N²) neighbors. **Thousands of images may be slow or crash**, especially on low-memory devices.
- CLIP weights and browser library require an internet connection on first use. Dependencies are fetched from public CDNs. A student shouldn't assume fully offline operation.
- Opening the generated map uses a blob URL. Popup blocking may prevent a new tab; use **Save interactive HTML**.
- This prototype has not been end-to-end tested in a live browser or published to your repository. Test in Chrome/Edge/Firefox with a small dataset, and then test performance on student devices, before recording the tutorial.
- The original map viewer saves annotations using browser localStorage; **always export annotated CSV/JSON** to keep edits permanently. A standalone saved map HTML includes initial image data but not later localStorage annotations until exported separately.
- For formal research reproducibility, record model/version and method settings and do not treat online/browser map output as automatically identical to desktop output.
