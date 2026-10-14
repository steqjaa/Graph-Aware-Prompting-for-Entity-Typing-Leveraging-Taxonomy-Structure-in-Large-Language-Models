# TUM thesis — editable LaTeX project

## Overleaf
1. Create a new project using **Upload Project** and upload this ZIP.
2. Set the main document to `main.tex` and compiler to **pdfLaTeX**.
3. Recompile. No Biber run, shell escape, Python, or result archives are required.

## Finish the cover metadata
Edit `metadata.tex`. The author, English title/subtitle, and submission date are carried over from the revised thesis. The German title, supervisor, advisor, and signing location are explicitly marked for completion. The thesis type is kept as “Bachelor’s Thesis”; amend the degree wording if required.
The template's declaration is included unsigned. Optional acknowledgements and a German abstract are not added because they were absent from the supplied thesis.

## Files
- `main.tex`: template structure and chapter order.
- `settings.tex`: typography and packages.
- `chapters/`: all ten chapters and three appendices, fully editable.
- `pages/`: cover, title, declaration, abstract, and reading conventions.
- `bibliography.tex`: full author-year references as editable `bibitem` entries; matching citations use `citet`/`citep`. This deliberately preserves the supplied reference wording without re-inferring bibliographic fields.
- `figures/`: the original numerical charts and vector pipeline diagram.
- `logos/`: template artwork.
- `main.pdf`: compiled preview.

The thesis prose, numerical results, table contents, and complete prompt texts are preserved from the latest revised PDF. Changes are limited to LaTeX/template formatting, automatic numbering/references, and the template's front matter. Table A.2 and A.3 retain their a/b panels.

Local compilation: `latexmk -pdf main.tex` (or run `pdflatex main.tex` three times).
See `TEMPLATE_LICENSE.md` for attribution and template licensing.
