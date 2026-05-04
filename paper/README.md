# Paper Build Guide

This folder contains the LaTeX source for the paper. The paper is split into section files so multiple people can edit different parts with fewer Git conflicts.

## Folder Structure

```text
paper/
  main.tex                  Main LaTeX file. Compile this.
  refs.bib                  Shared bibliography file.
  sections/                 Paper content split by section.
  figures/                  Images used by \includegraphics.
  build/                    Generated PDF and temporary files. Do not commit.
  latexmkrc                 Build configuration.
  Makefile                  Short build commands.
  Dockerfile                Docker image for paper compilation.
  docker-compose.yml        Docker command wrapper.
```

## Required Setup

Use Docker for paper builds. This keeps every collaborator on the same TeX Live version and package set.

Install Docker Desktop, then check that Docker works:

```sh
docker --version
```

You do not need to install TinyTeX, MacTeX, MiKTeX, or TeX Live locally.

## Compile the Paper with Docker

From this `paper/` directory:

```sh
make paper
```

The generated PDF will be:

```text
build/main.pdf
```

The first run downloads/builds the TeX Live image, so it can take several minutes. After that, builds are faster.

To force a fresh rebuild:

```sh
make paper-clean
make paper
```

For live rebuilds while editing:

```sh
make paper-watch
```

To clean Docker-generated LaTeX files (keeps PDF):

```sh
make paper-clean
```

To remove everything including the generated PDF:

```sh
make paper-distclean
```

## Optional Local Build

Local LaTeX is optional. Use it only if you already have a full LaTeX distribution with `latexmk`.

Check local LaTeX:

```sh
latexmk -v
```

Local build commands:

```sh
make
make watch
make clean
make distclean
```

## VS Code Preview

Install the VS Code extension **LaTeX Workshop** by James Yu.

Useful commands:

```text
Cmd + Option + B    Build LaTeX project
Cmd + Option + V    View PDF
```

If VS Code asks for a recipe, choose `latexmk`.

The LaTeX Workshop preview opens the same PDF as `build/main.pdf`, but it refreshes inside VS Code and can jump between the PDF and the `.tex` source.

## How to Edit

- Put document-level setup only in `main.tex`: title, authors, packages, and `\input{...}` lines.
- Put actual paper text in the files under `sections/`.
- Put images in `figures/`.
- Put references in `refs.bib`.
- Do not edit generated files inside `build/`.

### Template-Specific Fields in `main.tex`

The paper uses the Elsevier CAS double-column template (`cas-dc`). Key fields:

| Field | Purpose |
|---|---|
| `\cormark[N]` / `\cortext[N]` | Mark corresponding author and set the footnote text |
| `\fnmark[N]` / `\fntext[N]` | Author footnotes (used here for the acknowledgment) |
| `\credit{...}` | CRediT taxonomy contribution per author |
| `\begin{highlights}` | Research highlights shown before the abstract |
| `\printcredits` | Prints the credit table — place before `\bibliography` |

When adding a new section file, add it to `main.tex` with:

```tex
\input{sections/folder_name/file_name}
```

Do not include the `.tex` extension in `\input`.

## Git Rules

Commit these files:

```text
main.tex
latexmkrc
Makefile
Dockerfile
docker-compose.yml
refs.bib
sections/**/*.tex
figures/*
cas-dc.cls
cas-common.sty
cas-model2-names.bst
```

Do not commit generated files:

```text
build/
*.aux
*.log
*.out
*.toc
*.bbl
*.blg
*.fls
*.fdb_latexmk
*.synctex.gz
*.pdf
*.bak*
```

The local `.gitignore` in this folder already ignores these.

## Current Warnings

The paper may compile with warnings. These are known:

- `File figures/... not found`: some figure image files are not added yet.
- `Overfull \hbox`: some lines, equations, table rows, or code-style names are too wide.
- `Warning--to sort, need author or key in scholareval` (and `rinobench`, `opennovelty`): these three bib entries are placeholder citations for papers not yet publicly available. They have no author field. Fix when the actual references are known.
- `Warning--empty pages in ...`: several conference/workshop entries in `refs.bib` are missing a `pages` field. Add page numbers before final submission.

These warnings do not prevent writing. Before final submission, add the missing figures, resolve the placeholder references, add missing page numbers, and fix large overfull boxes.
