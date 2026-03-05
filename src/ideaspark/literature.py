"""Literature monitoring — pull recent papers from bioRxiv, arXiv, PubMed."""

import json
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path

import requests

from src.config import PROJECT_ROOT

logger = logging.getLogger(__name__)

CACHE_DIR = PROJECT_ROOT / "data" / "ideaspark" / "literature_cache"


class LiteratureMonitor:
    """Fetch and cache recent preprints and papers from multiple sources."""

    def __init__(self, window_days: int = 30):
        self.window_days = window_days
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # ── bioRxiv ───────────────────────────────────────────────────────

    def fetch_biorxiv(self, categories: list[str] | None = None, limit: int = 100) -> list[dict]:
        """Fetch recent bioRxiv preprints via the API.

        Categories: genomics, bioinformatics, cancer_biology, systems_biology,
                    molecular_biology, cell_biology, genetics
        """
        if categories is None:
            categories = [
                "genomics", "bioinformatics", "cancer_biology",
                "systems_biology", "molecular_biology",
            ]

        end = datetime.now()
        start = end - timedelta(days=self.window_days)
        date_range = f"{start.strftime('%Y-%m-%d')}/{end.strftime('%Y-%m-%d')}"

        papers = []
        for cat in categories:
            url = f"https://api.biorxiv.org/details/biorxiv/{date_range}/0/50"
            try:
                resp = requests.get(url, timeout=30)
                resp.raise_for_status()
                data = resp.json()
                for item in data.get("collection", []):
                    if item.get("category", "").lower().replace(" ", "_") == cat or not categories:
                        papers.append({
                            "source": "biorxiv",
                            "title": item.get("title", ""),
                            "abstract": item.get("abstract", ""),
                            "authors": item.get("authors", ""),
                            "doi": item.get("doi", ""),
                            "date": item.get("date", ""),
                            "category": item.get("category", ""),
                            "url": f"https://doi.org/{item.get('doi', '')}",
                        })
                time.sleep(1)  # rate limit
            except Exception as e:
                logger.warning(f"bioRxiv fetch failed for {cat}: {e}")

        logger.info(f"bioRxiv: fetched {len(papers)} papers")
        return papers[:limit]

    # ── arXiv ─────────────────────────────────────────────────────────

    def fetch_arxiv(self, categories: list[str] | None = None, limit: int = 100) -> list[dict]:
        """Fetch recent arXiv preprints via the Atom API.

        Categories: cs.LG, cs.AI, cs.CL, q-bio.GN, q-bio.QM
        """
        if categories is None:
            categories = ["cs.LG", "cs.AI", "cs.CL", "q-bio.GN", "q-bio.QM"]

        cat_query = "+OR+".join([f"cat:{c}" for c in categories])
        url = (
            f"http://export.arxiv.org/api/query"
            f"?search_query={cat_query}"
            f"&sortBy=submittedDate&sortOrder=descending"
            f"&max_results={limit}"
        )

        papers = []
        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()

            # Simple XML parsing (avoid heavy dependency)
            import xml.etree.ElementTree as ET
            root = ET.fromstring(resp.text)
            ns = {"atom": "http://www.w3.org/2005/Atom"}

            for entry in root.findall("atom:entry", ns):
                title = entry.findtext("atom:title", "", ns).strip().replace("\n", " ")
                abstract = entry.findtext("atom:summary", "", ns).strip().replace("\n", " ")
                authors = [a.findtext("atom:name", "", ns) for a in entry.findall("atom:author", ns)]
                published = entry.findtext("atom:published", "", ns)[:10]
                link = ""
                for l in entry.findall("atom:link", ns):
                    if l.get("type") == "text/html":
                        link = l.get("href", "")
                        break

                arxiv_cats = [c.get("term", "") for c in entry.findall("atom:category", ns)]

                papers.append({
                    "source": "arxiv",
                    "title": title,
                    "abstract": abstract,
                    "authors": ", ".join(authors),
                    "date": published,
                    "category": ", ".join(arxiv_cats[:3]),
                    "url": link,
                })

        except Exception as e:
            logger.warning(f"arXiv fetch failed: {e}")

        logger.info(f"arXiv: fetched {len(papers)} papers")
        return papers[:limit]

    # ── PubMed ────────────────────────────────────────────────────────

    def fetch_pubmed(self, queries: list[str] | None = None, limit: int = 50) -> list[dict]:
        """Fetch recent PubMed articles via E-utilities.

        Default queries target cancer + AI/ML intersections.
        """
        if queries is None:
            queries = [
                "(foundation model OR language model) AND (biology OR genomics OR protein)",
                "(single cell) AND (deep learning OR neural network OR foundation model)",
                "(RNA) AND (machine learning OR deep learning) AND (structure OR therapeutics)",
                "(perturbation OR CRISPR screen) AND (machine learning OR prediction)",
            ]

        end = datetime.now()
        start = end - timedelta(days=self.window_days)
        mindate = start.strftime("%Y/%m/%d")
        maxdate = end.strftime("%Y/%m/%d")

        all_pmids = set()
        for query in queries:
            try:
                search_url = (
                    "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
                    f"?db=pubmed&retmode=json&retmax={limit}"
                    f"&datetype=pdat&mindate={mindate}&maxdate={maxdate}"
                    f"&term={requests.utils.quote(query)}"
                )
                resp = requests.get(search_url, timeout=30)
                resp.raise_for_status()
                data = resp.json()
                pmids = data.get("esearchresult", {}).get("idlist", [])
                all_pmids.update(pmids)
                time.sleep(0.5)
            except Exception as e:
                logger.warning(f"PubMed search failed for query '{query[:40]}...': {e}")

        if not all_pmids:
            return []

        # Fetch summaries
        papers = []
        pmid_list = list(all_pmids)[:limit]
        try:
            fetch_url = (
                "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
                f"?db=pubmed&retmode=json&id={','.join(pmid_list)}"
            )
            resp = requests.get(fetch_url, timeout=30)
            resp.raise_for_status()
            data = resp.json()

            for pmid in pmid_list:
                info = data.get("result", {}).get(pmid, {})
                if not info or pmid == "uids":
                    continue
                authors = [a.get("name", "") for a in info.get("authors", [])]
                papers.append({
                    "source": "pubmed",
                    "title": info.get("title", ""),
                    "abstract": "",  # summaries don't include abstracts; fetched separately if needed
                    "authors": ", ".join(authors[:5]),
                    "date": info.get("pubdate", ""),
                    "journal": info.get("fulljournalname", ""),
                    "pmid": pmid,
                    "doi": next(
                        (aid.get("value", "") for aid in info.get("articleids", []) if aid.get("idtype") == "doi"),
                        "",
                    ),
                    "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                })
        except Exception as e:
            logger.warning(f"PubMed fetch failed: {e}")

        logger.info(f"PubMed: fetched {len(papers)} papers")
        return papers

    # ── combined fetch ────────────────────────────────────────────────

    def fetch_all(self, limit_per_source: int = 50) -> list[dict]:
        """Fetch from all sources and return combined list."""
        papers = []
        papers.extend(self.fetch_biorxiv(limit=limit_per_source))
        papers.extend(self.fetch_arxiv(limit=limit_per_source))
        papers.extend(self.fetch_pubmed(limit=limit_per_source))
        logger.info(f"Total new literature: {len(papers)} papers")
        return papers

    # ── caching ───────────────────────────────────────────────────────

    def cache_papers(self, papers: list[dict], date_str: str | None = None):
        """Save fetched papers to daily cache."""
        if date_str is None:
            date_str = datetime.now().strftime("%Y-%m-%d")
        path = CACHE_DIR / f"{date_str}.json"
        with open(path, "w") as f:
            json.dump(papers, f, indent=2)
        logger.info(f"Cached {len(papers)} papers to {path}")

    def load_cache(self, date_str: str | None = None) -> list[dict]:
        """Load cached papers for a given date."""
        if date_str is None:
            date_str = datetime.now().strftime("%Y-%m-%d")
        path = CACHE_DIR / f"{date_str}.json"
        if path.exists():
            with open(path) as f:
                return json.load(f)
        return []

    def has_today_cache(self) -> bool:
        """Check if we already fetched today."""
        path = CACHE_DIR / f"{datetime.now().strftime('%Y-%m-%d')}.json"
        return path.exists()
