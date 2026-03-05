"""Deep research module — multi-turn Claude exploration of unfamiliar domains.

Before generating an idea, the agent conducts a research session:
1. Given a theme (with its unfamiliar domain), generate targeted search queries
2. Pull papers from PubMed, bioRxiv, arXiv for the UNFAMILIAR domain
3. Have Claude synthesize findings into a research brief
4. Feed the brief + Hani's anchor papers into the idea generator

This makes each idea grounded in real, current work from the target domain.
"""

import json
import logging
import time
from datetime import datetime, timedelta

import anthropic
import requests

from src.config import ANTHROPIC_API_KEY, AGENT_MODEL

logger = logging.getLogger(__name__)

# ── Search helpers ───────────────────────────────────────────────────

def _pubmed_search(query: str, max_results: int = 20, days: int = 90) -> list[dict]:
    """Search PubMed for papers matching query within date window."""
    end = datetime.now()
    start = end - timedelta(days=days)
    try:
        search_url = (
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
            f"?db=pubmed&retmode=json&retmax={max_results}"
            f"&datetype=pdat&mindate={start.strftime('%Y/%m/%d')}"
            f"&maxdate={end.strftime('%Y/%m/%d')}"
            f"&term={requests.utils.quote(query)}"
        )
        resp = requests.get(search_url, timeout=30)
        resp.raise_for_status()
        pmids = resp.json().get("esearchresult", {}).get("idlist", [])
        if not pmids:
            return []
        time.sleep(0.4)

        fetch_url = (
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
            f"?db=pubmed&retmode=json&id={','.join(pmids)}"
        )
        resp = requests.get(fetch_url, timeout=30)
        resp.raise_for_status()
        data = resp.json().get("result", {})

        papers = []
        for pmid in pmids:
            info = data.get(pmid, {})
            if not info or pmid == "uids":
                continue
            authors = [a.get("name", "") for a in info.get("authors", [])]
            papers.append({
                "source": "pubmed",
                "title": info.get("title", ""),
                "authors": ", ".join(authors[:3]),
                "date": info.get("pubdate", ""),
                "journal": info.get("fulljournalname", ""),
                "pmid": pmid,
            })
        return papers
    except Exception as e:
        logger.warning(f"PubMed deep search failed: {e}")
        return []


def _biorxiv_search(query_terms: list[str], max_results: int = 20, days: int = 60) -> list[dict]:
    """Fetch recent bioRxiv preprints and keyword-filter for query terms."""
    end = datetime.now()
    start = end - timedelta(days=days)
    date_range = f"{start.strftime('%Y-%m-%d')}/{end.strftime('%Y-%m-%d')}"

    papers = []
    try:
        url = f"https://api.biorxiv.org/details/biorxiv/{date_range}/0/100"
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        collection = resp.json().get("collection", [])

        terms_lower = [t.lower() for t in query_terms]
        for item in collection:
            text = f"{item.get('title', '')} {item.get('abstract', '')}".lower()
            if any(t in text for t in terms_lower):
                papers.append({
                    "source": "biorxiv",
                    "title": item.get("title", ""),
                    "abstract": item.get("abstract", "")[:300],
                    "authors": item.get("authors", ""),
                    "date": item.get("date", ""),
                    "category": item.get("category", ""),
                })
    except Exception as e:
        logger.warning(f"bioRxiv deep search failed: {e}")

    return papers[:max_results]


# ── Deep research pipeline ───────────────────────────────────────────

QUERY_GEN_PROMPT = """You are a research librarian helping a computational biologist explore an UNFAMILIAR domain.

Theme for today's exploration: {theme_name}
Description: {theme_description}

The researcher (Hani Goodarzi) is an expert in: RNA biology, cancer genomics, AI/ML foundation models, single-cell omics, liquid biopsy.

His ANCHOR for today is one of these papers/capabilities:
{anchor_summary}

Your task: generate 4-6 TARGETED PubMed search queries that will find the most interesting recent work in the UNFAMILIAR parts of this theme. The queries should:
1. Focus on the destination domain (NOT on cancer or RNA biology — Hani already knows that)
2. Include methodological terms that would catch papers where Hani's tools could apply
3. Be specific enough to return high-quality results (not thousands of generic hits)
4. Cover different angles of the unfamiliar domain

Also generate 2-3 keyword lists for bioRxiv filtering (bioRxiv doesn't support complex queries).

Return ONLY a JSON object:
```json
{{
  "pubmed_queries": ["query1", "query2", ...],
  "biorxiv_keywords": [["term1", "term2"], ["term3", "term4"], ...]
}}
```"""

SYNTHESIS_PROMPT = """You are a research analyst preparing a briefing for a computational biologist who is EXPLORING OUTSIDE HIS COMFORT ZONE.

The researcher (Hani Goodarzi) is expert in: RNA biology, cancer genomics, AI/ML foundation models, single-cell omics, liquid biopsy. He is NOT an expert in the domain below.

Today's theme: {theme_name}
{theme_description}

Here are papers found in the UNFAMILIAR domain:

{papers_formatted}

Your task: write a research brief (500-800 words) that:
1. Identifies the 3-4 most exciting open problems or recent breakthroughs in this domain
2. For each, explains WHY someone with Hani's specific toolkit might have a unique angle
3. Highlights any paper that seems ripe for cross-pollination with genomic/RNA/single-cell AI methods
4. Notes specific datasets, challenges, or collaborator types in this domain
5. Is honest about what you DON'T know — flag gaps where deeper reading would help

Write it as a crisp briefing, not a lit review. Be opinionated about what's most promising."""


class DeepResearcher:
    """Conducts multi-step research into unfamiliar domains before idea generation."""

    def __init__(self):
        self.client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    def research_theme(
        self,
        theme: dict,
        anchor_papers: list[dict],
    ) -> dict:
        """Run deep research for a theme. Returns research brief + discovered papers.

        Steps:
        1. Generate targeted search queries via Claude
        2. Execute searches against PubMed + bioRxiv
        3. Synthesize findings into a research brief
        """
        logger.info(f"Deep research: {theme['name']}")

        # Format anchor papers
        anchor_summary = "\n".join(
            f"- {p.get('title', '')} ({p.get('journal', '')}, {p.get('year', '')})"
            for p in anchor_papers[:5]
        )

        # Step 1: Generate search queries
        queries = self._generate_queries(theme, anchor_summary)
        if not queries:
            logger.warning("Query generation failed, using theme keywords as fallback")
            queries = {
                "pubmed_queries": [theme["query"]],
                "biorxiv_keywords": [theme["query"].split()[:3]],
            }

        # Step 2: Execute searches
        all_papers = []

        for q in queries.get("pubmed_queries", [])[:5]:
            papers = _pubmed_search(q, max_results=15, days=90)
            all_papers.extend(papers)
            time.sleep(0.5)

        for kw_list in queries.get("biorxiv_keywords", [])[:3]:
            papers = _biorxiv_search(kw_list, max_results=10, days=60)
            all_papers.extend(papers)
            time.sleep(0.5)

        # Deduplicate by title similarity (simple)
        seen_titles = set()
        unique_papers = []
        for p in all_papers:
            title_key = p["title"].lower()[:60]
            if title_key not in seen_titles:
                seen_titles.add(title_key)
                unique_papers.append(p)

        logger.info(f"Deep research found {len(unique_papers)} unique papers")

        # Step 3: Synthesize
        brief = self._synthesize(theme, unique_papers)

        return {
            "research_brief": brief,
            "papers": unique_papers,
            "queries_used": queries,
        }

    def _generate_queries(self, theme: dict, anchor_summary: str) -> dict | None:
        """Ask Claude to generate targeted search queries."""
        prompt = QUERY_GEN_PROMPT.format(
            theme_name=theme["name"],
            theme_description=theme["description"],
            anchor_summary=anchor_summary,
        )
        try:
            response = self.client.messages.create(
                model=AGENT_MODEL,
                max_tokens=800,
                messages=[{"role": "user", "content": prompt}],
            )
            text = response.content[0].text

            # Extract JSON
            json_start = text.find("{")
            json_end = text.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                return json.loads(text[json_start:json_end])
        except Exception as e:
            logger.warning(f"Query generation failed: {e}")
        return None

    def _synthesize(self, theme: dict, papers: list[dict]) -> str:
        """Have Claude synthesize discovered papers into a research brief."""
        if not papers:
            return "(No papers found in deep research — falling back to broad theme description.)"

        papers_formatted = "\n\n".join(
            f"[{p.get('source', '')}] {p.get('title', '')}\n"
            f"  Authors: {p.get('authors', 'N/A')}\n"
            f"  Date: {p.get('date', 'N/A')} | Journal: {p.get('journal', p.get('category', 'N/A'))}\n"
            + (f"  Abstract: {p['abstract'][:250]}..." if p.get("abstract") else "")
            for p in papers[:25]  # cap at 25 to stay within context
        )

        prompt = SYNTHESIS_PROMPT.format(
            theme_name=theme["name"],
            theme_description=theme["description"],
            papers_formatted=papers_formatted,
        )

        try:
            response = self.client.messages.create(
                model=AGENT_MODEL,
                max_tokens=1500,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text
        except Exception as e:
            logger.warning(f"Synthesis failed: {e}")
            return "(Synthesis failed — using raw paper list.)"
