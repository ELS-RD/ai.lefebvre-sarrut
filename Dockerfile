FROM ghcr.io/els-rd/mkdocs-material-insiders:4.30.2

# lightbox plugin
RUN pip install --no-cache-dir \
  pip install mkdocs-glightbox