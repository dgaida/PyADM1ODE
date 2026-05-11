// MathJax configuration for the pymdownx.arithmatex (generic) extension.
// Inline math is delimited with \( ... \) and display math with \[ ... \] by
// the markdown extension, but we also keep the $...$ / $$...$$ delimiters so
// that math blocks written directly in markdown render too.
window.MathJax = {
  tex: {
    inlineMath: [["\\(", "\\)"], ["$", "$"]],
    displayMath: [["\\[", "\\]"], ["$$", "$$"]],
    processEscapes: true,
    processEnvironments: true,
  },
  options: {
    ignoreHtmlClass: ".*|",
    processHtmlClass: "arithmatex",
  },
};

// Re-typeset math on every Material-for-MkDocs instant-navigation page load.
document$.subscribe(() => {
  if (typeof MathJax !== "undefined" && MathJax.typesetPromise) {
    MathJax.typesetPromise();
  }
});
