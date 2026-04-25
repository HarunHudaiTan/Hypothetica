/** UI strings keyed by `PaperDetail.source` / adapter id */

export function sourceLinkLabel(source: string): string {
  switch (source) {
    case "arxiv":
      return "arXiv";
    case "google_patents":
      return "Google Patents";
    case "github":
      return "GitHub";
    default:
      return source.replace(/_/g, " ");
  }
}

export function evidenceTableHeading(source: string | undefined): {
  title: string;
  itemWord: string;
} {
  if (source === "github") {
    return { title: "Analyzed Repositories", itemWord: "repo" };
  }
  if (source === "google_patents") {
    return { title: "Analyzed Patents", itemWord: "patent" };
  }
  return { title: "Analyzed Papers", itemWord: "paper" };
}

export function evidenceStatsLabels(source: string | undefined): {
  analyzed: string;
  fetched: string;
} {
  if (source === "github") {
    return { analyzed: "Repos Analyzed", fetched: "Repos Fetched" };
  }
  if (source === "google_patents") {
    return { analyzed: "Patents Analyzed", fetched: "Patents Fetched" };
  }
  return { analyzed: "Papers Analyzed", fetched: "Papers Fetched" };
}
