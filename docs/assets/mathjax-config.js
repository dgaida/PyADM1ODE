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

// Re-typeset math on every Material-for-MkDocs page load.  With
// navigation.instant disabled, MathJax auto-typesets on first load anyway,
// but this hook keeps things consistent if instant navigation is re-enabled
// in the future.  Guard against `document$` not being exposed.
if (typeof document$ !== "undefined" && document$.subscribe) {
  document$.subscribe(() => {
    if (typeof MathJax !== "undefined" && MathJax.typesetPromise) {
      MathJax.typesetPromise();
    }
  });
}
