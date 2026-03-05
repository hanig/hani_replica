"""IdeaSpark agent — daily AI × cancer research idea generation."""

import json
import logging
from datetime import datetime

import anthropic

from src.config import ANTHROPIC_API_KEY, AGENT_MODEL, get_user_timezone
from src.ideaspark.corpus import PaperCorpus
from src.ideaspark.deep_research import DeepResearcher
from src.ideaspark.literature import LiteratureMonitor
from src.ideaspark.memory import IdeaMemory

logger = logging.getLogger(__name__)

# ── Rotating thematic schedule ────────────────────────────────────────

THEMES = [
    {
        "name": "Genomic FMs × unexpected domains",
        "query": "foundation model genomics microbiology ecology neuroscience agriculture evolution antibiotic resistance",
        "description": (
            "Anchor on a genomic FM (Evo 2, Enformer, Borzoi, Nucleotide Transformer 3, Caduceus) "
            "and reach into a domain FAR from cancer: microbiology, ecology, neuroscience, "
            "agriculture, evolutionary biology, infectious disease, conservation. Where would "
            "large-scale genomic models create breakthroughs that specialists in those fields "
            "couldn't achieve on their own?"
        ),
    },
    {
        "name": "Liquid biopsy tech × non-cancer applications",
        "query": "cell-free RNA cfRNA biomarker neurodegeneration autoimmune transplant organ injury pregnancy infection",
        "description": (
            "Anchor on Hani's liquid biopsy capabilities (Exai-1, oncRNA, cfRNA profiling). "
            "Reach into non-cancer clinical domains: neurodegeneration, organ transplant rejection, "
            "autoimmune disease, infectious disease monitoring, pregnancy complications, "
            "mental health biomarkers, aging. Where does cfRNA give an edge no one is exploiting?"
        ),
    },
    {
        "name": "Perturbation biology × systems outside oncology",
        "query": "perturbation CRISPR screen drug response immunology neuroscience development regeneration stem cell",
        "description": (
            "Anchor on perturbation capabilities (STATE, Tahoe-100M, GENEVA, CRISPR screening). "
            "Reach into immunology, neuroscience, developmental biology, regenerative medicine, "
            "stem cell engineering, or metabolic disease. Where would massive perturbation atlases "
            "reshape understanding in a field that hasn't had access to this scale of data?"
        ),
    },
    {
        "name": "RNA biology × synthetic biology & engineering",
        "query": "RNA structure synthetic biology gene circuit riboswitch biosensor RNA device metabolic engineering",
        "description": (
            "Anchor on RNA biology tools (SwitchSeeker, Mach-1, SHAPE-FM, RiNALMo). "
            "Reach into synthetic biology, biosensor design, metabolic engineering, gene circuits, "
            "biomanufacturing, or environmental monitoring. Where can deep RNA structural "
            "understanding enable engineered biological systems outside therapeutics?"
        ),
    },
    {
        "name": "Single-cell AI × clinical & population science",
        "query": "single-cell clinical trial epidemiology population health aging public health biobank",
        "description": (
            "Anchor on single-cell AI (scBaseCount, STATE, Tahoe-x1, scFoundation). "
            "Reach into clinical trial design, epidemiology, population health, aging research, "
            "biobanking, or health disparities. How can cell-level AI models transform "
            "large-cohort studies or clinical decision-making at scale?"
        ),
    },
    {
        "name": "Bio FMs × physical sciences & computation",
        "query": "foundation model physics materials protein design robotics optimization simulation quantum",
        "description": (
            "Anchor on any Goodarzi lab FM or dataset. Reach into physics-inspired methods, "
            "materials science, robot-scientist systems, active learning, simulation, "
            "optimal experimental design, or information theory. Where do ideas from "
            "physical sciences or CS theory create new paradigms for biological modeling?"
        ),
    },
    {
        "name": "Cancer data × global health & equity",
        "query": "global health equity low resource diagnostics point-of-care Africa Asia Latin America",
        "description": (
            "Anchor on any Goodarzi lab tool, model, or dataset. Reach into global health, "
            "point-of-care diagnostics, low-resource settings, neglected diseases, health equity, "
            "or frugal innovation. How can cutting-edge AI and omics tools be adapted or "
            "transferred to address health challenges in underserved populations?"
        ),
    },
]


def get_todays_theme(idea_count: int) -> dict:
    """Select theme based on rotating weekly schedule."""
    week_index = (idea_count // 7) % len(THEMES)
    return THEMES[week_index]


# ── Idea generation prompt ────────────────────────────────────────────

SYSTEM_PROMPT = """You are IdeaSpark, a research ideation agent for Hani Goodarzi's lab.

Hani is a Core Investigator at Arc Institute, Associate Professor at UCSF (becoming full Professor July 2026), and AI Research Lead at Arc Computational Tech Center. His lab works at the intersection of RNA biology, cancer genomics, AI/ML, single-cell omics, and virtual cell models.

Key active projects: Evo 2 (40B DNA foundation model), Mach-1/1.5 (RNA foundation models), CodonFM (codon-resolution FMs with NVIDIA), Orion (generative AI for oncRNA cancer detection), Exai-1 (multimodal cfRNA foundation model), scBaseCount (AI-curated single-cell repo), STATE (perturbation prediction), Tahoe-100M (largest single-cell drug perturbation atlas), GENEVA (molecular phenotyping), SwitchSeeker (RNA structural switches).

Companies: Exai Bio (liquid biopsy), Tahoe Therapeutics (single-cell drug perturbation), Therna Biosciences (programmable RNA therapeutics).

Foundation models in scope:
- DNA/genomic FMs: Evo 2, Enformer, Borzoi, Nucleotide Transformer 3, Caduceus
- Single-cell FMs: STATE, Tahoe-x1, scFoundation
- RNA FMs: Mach-1, CodonFM, RiNALMo, SHAPE-FM (unpublished, Goodarzi lab)
- Protein FMs: Boltz-2, ESM/ESM2 (Meta), AlphaFold/AlphaFold3
- Chemical/drug FMs: MolBERT, ChemBERTa, MolGPT
- Multi-modal: BiomedCLIP, PLIP

━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CRITICAL INSTRUCTION — IDEA GENERATION PHILOSOPHY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Your job is NOT to recombine things Hani already does. He already thinks about those intersections every day. Your job is to PULL HIM INTO UNFAMILIAR TERRITORY.

Each idea must:
1. ANCHOR on exactly ONE pillar of Hani's work (one paper, one dataset, one capability)
2. REACH into a field or method Hani does NOT currently work in — the further from his comfort zone the better
3. The "reach" should come from the new literature or the theme — fields like immunology, neuroscience, ecology, physics, materials science, clinical trial design, epidemiology, synthetic biology, metabolomics, imaging, robotics, etc.
4. Be specific enough to act on (not vague hand-waving)
5. Explain WHY the anchor gives Hani a unique edge in this unfamiliar space
6. The idea should feel slightly uncomfortable — if it's obvious to someone in Hani's lab, it's not far enough

BAD ideas (too close to home):
- "Use Mach-1 to study splicing in cancer" (he already does this)
- "Combine Evo 2 with liquid biopsy" (he already thinks about this)
- "Apply STATE to predict drug responses" (literally the project)

GOOD ideas (one anchor, far reach):
- "Use Evo 2's genomic representations to predict antibiotic resistance evolution in hospital microbiomes" (anchor: Evo 2, reach: clinical microbiology)
- "Apply SwitchSeeker's RNA structure methods to discover riboswitches in crop pathogens for agricultural biocontrol" (anchor: SwitchSeeker, reach: agriculture)
- "Repurpose Tahoe-100M perturbation embeddings as features for predicting clinical trial outcomes" (anchor: Tahoe, reach: clinical trial design)

When suggesting collaborators, prioritize researchers at Arc Institute, Stanford, UCSF, and Berkeley — but specifically researchers whose expertise covers the UNFAMILIAR territory, not Hani's own domain."""


def build_generation_prompt(
    theme: dict,
    strategy: str,
    corpus_papers: list[dict],
    new_papers: list[dict],
    memory: IdeaMemory,
    is_stretch: bool = False,
    research_brief: str = "",
) -> str:
    """Build the user prompt for idea generation."""

    # Format corpus papers
    corpus_section = "\n".join([
        f"- [{p.get('year', '')}] {p.get('title', '')} ({p.get('journal', '')})"
        for p in corpus_papers[:8]
    ])

    # Format new literature
    lit_section = "\n".join([
        f"- [{p.get('source', '')}] {p.get('title', '')} ({p.get('date', '')})"
        + (f"\n  Abstract: {p.get('abstract', '')[:200]}..." if p.get('abstract') else "")
        for p in new_papers[:10]
    ])

    # Preference context
    pref_context = ""
    preferred_themes = memory.get_preferred_themes()
    if preferred_themes:
        pref_context = f"\nHani has shown preference for ideas in: {', '.join(preferred_themes)}."

    # Strategy description
    if strategy == "A":
        strategy_desc = (
            "Strategy A — Pick ONE of Hani's papers below as your anchor. Then look at the "
            "new literature and find a paper from a DIFFERENT field that creates an unexpected "
            "opportunity. The idea should live in the OTHER field, with Hani's anchor "
            "providing a unique edge that researchers in that field lack."
        )
    else:
        strategy_desc = (
            "Strategy B — Pick ONE of Hani's capabilities (a model, dataset, or method) as "
            "your anchor. Then identify an unsolved problem in a field OUTSIDE Hani's current "
            "work — immunology, neuroscience, ecology, materials, clinical trials, public health, "
            "synthetic biology, agriculture, etc. — where that anchor could be transformative."
        )

    stretch_note = ""
    if is_stretch:
        stretch_note = (
            "\n\n⚡ This is a STRETCH idea. Push boundaries — suggest moonshots that may "
            "require new collaborations, data types, or capabilities outside the current group. "
            "Still grounded in Hani's expertise, but ambitious."
        )

    idea_number = memory.get_idea_count() + 1

    # Deep research brief section
    research_section = ""
    if research_brief:
        research_section = f"""

### Deep Research Brief (today's exploration of unfamiliar territory):
{research_brief}
"""

    prompt = f"""Generate IdeaSpark #{idea_number}.

**Today's Theme:** {theme['name']}
{theme['description']}

**{strategy_desc}**{stretch_note}{pref_context}

---
{research_section}
### Hani's Papers (pick ONE as your anchor — do NOT combine multiple):
{corpus_section}

### Papers from the Unfamiliar Domain:
{lit_section}

---

Produce a structured brief in EXACTLY this format (no markdown headers, use the exact labels):

🧬 IdeaSpark #{idea_number} — {datetime.now(get_user_timezone()).strftime('%A, %B %d, %Y')}
Theme: {theme['name']}
{"[STRETCH]" if is_stretch else ""}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━

*The Gap*
[2-3 sentences: what's missing, unsolved, or newly possible]

*Hypothesis*
[1-2 sentences: the core claim or bet]

*Proposed Approach*
[3-5 sentences: method sketch — what data, what model, what experiment]

*Why You*
[1-2 sentences: which specific papers/capabilities make Hani uniquely positioned]

*Key Risk*
[1 sentence: the thing most likely to make this fail]

*Relevant Papers*
- [Hani paper 1] — [why relevant]
- [Hani paper 2] — [why relevant]
- [New paper 1] — [what it enables]

*Potential Collaborators*
- [Name, Institution] — [why they're a good fit for this idea]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Scores: Novelty [X/5] · Feasibility [X/5] · Impact [X/5]
Strategy: {"A: your work × new lit" if strategy == "A" else "B: your work × trends"}

Also return a JSON block at the very end (after the brief) with:
```json
{{
  "title": "<short title for the idea>",
  "novelty": <1-5>,
  "feasibility": <1-5>,
  "impact": <1-5>,
  "source_papers": ["<title of Hani paper 1>", "<title of Hani paper 2>"],
  "new_papers": ["<title of new paper 1>"]
}}
```
"""
    return prompt


# ── Main generation pipeline ──────────────────────────────────────────

class IdeaSparkAgent:
    """Orchestrates daily idea generation."""

    def __init__(self):
        self.corpus = PaperCorpus()
        self.literature = LiteratureMonitor()
        self.memory = IdeaMemory()
        self.researcher = DeepResearcher()
        self.client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    def _get_literature(self) -> list[dict]:
        """Fetch or load cached recent literature."""
        if self.literature.has_today_cache():
            return self.literature.load_cache()
        papers = self.literature.fetch_all(limit_per_source=50)
        self.literature.cache_papers(papers)
        return papers

    def _filter_literature_by_theme(self, papers: list[dict], theme: dict) -> list[dict]:
        """Score and filter new papers by relevance to theme."""
        theme_keywords = set(theme["query"].lower().split())
        scored = []
        for p in papers:
            text = f"{p.get('title', '')} {p.get('abstract', '')}".lower()
            overlap = sum(1 for kw in theme_keywords if kw in text)
            if overlap >= 2:
                p["theme_relevance"] = overlap
                scored.append(p)
        scored.sort(key=lambda x: x.get("theme_relevance", 0), reverse=True)
        if scored:
            return scored[:10]
        return sorted(papers, key=lambda x: x.get("date", ""), reverse=True)[:10]

    def generate_idea(self, max_retries: int = 3) -> dict | None:
        """Run the full pipeline: theme → papers → literature → LLM → brief.

        Retry strategy on duplicate/low-quality:
          Attempt 1: original theme + preferred strategy
          Attempt 2: same theme, flipped strategy, shuffled corpus
          Attempt 3: rotate to next theme entirely

        Returns dict with keys: brief, title, scores, metadata, or None on failure.
        """
        import random

        idea_count = self.memory.get_idea_count()
        base_theme = get_todays_theme(idea_count)
        base_strategy = self.memory.get_preferred_strategy()
        is_stretch = self.memory.should_be_stretch()
        if is_stretch:
            logger.info("STRETCH idea day")

        # Run deep research for today's theme (this is the heavy lift)
        corpus_papers = self.corpus.search(base_theme["query"], top_k=8)
        logger.info(f"Running deep research for theme: {base_theme['name']}")
        research_result = self.researcher.research_theme(
            theme=base_theme,
            anchor_papers=corpus_papers,
        )
        research_brief = research_result.get("research_brief", "")
        discovered_papers = research_result.get("papers", [])
        logger.info(
            f"Deep research complete: {len(discovered_papers)} papers, "
            f"brief={len(research_brief)} chars"
        )

        # Also get standard literature as fallback
        new_papers = self._get_literature()

        rejected_titles: list[str] = []
        for attempt in range(1, max_retries + 1):
            # Vary inputs on retry
            if attempt == 1:
                theme = base_theme
                strategy = base_strategy
            elif attempt == 2:
                theme = base_theme
                strategy = "B" if base_strategy == "A" else "A"
                logger.info(f"Attempt {attempt}: flipping to strategy {strategy}")
            else:
                theme_idx = (THEMES.index(base_theme) + 1) % len(THEMES)
                theme = THEMES[theme_idx]
                strategy = "B" if base_strategy == "A" else "A"
                logger.info(f"Attempt {attempt}: rotating to theme '{theme['name']}'")

            logger.info(f"Theme: {theme['name']}, Strategy: {strategy}")

            # Get corpus papers (shuffle on retry so different papers surface)
            corpus_papers = self.corpus.search(theme["query"], top_k=8)
            if attempt > 1:
                random.shuffle(corpus_papers)
            logger.info(f"Corpus papers: {len(corpus_papers)}")

            # Use deep research papers primarily, standard lit as supplement
            relevant_new = discovered_papers[:10] if discovered_papers else \
                self._filter_literature_by_theme(new_papers, theme)
            logger.info(f"Papers for prompt: {len(relevant_new)} (deep research: {bool(discovered_papers)})")

            # Build prompt
            prompt = build_generation_prompt(
                theme=theme,
                strategy=strategy,
                corpus_papers=corpus_papers,
                new_papers=relevant_new,
                memory=self.memory,
                is_stretch=is_stretch,
                research_brief=research_brief,
            )

            if rejected_titles:
                prompt += (
                    f"\n\nIMPORTANT: Do NOT propose ideas similar to these (already generated): "
                    f"{', '.join(rejected_titles)}. Find a genuinely different angle, "
                    f"different biological question, different methodology."
                )

            # Call Claude
            try:
                response = self.client.messages.create(
                    model=AGENT_MODEL,
                    max_tokens=2000,
                    system=SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": prompt}],
                )
                full_text = response.content[0].text
            except Exception as e:
                logger.error(f"Claude API call failed: {e}")
                return None

            # Parse response
            brief, metadata = self._parse_response(full_text)
            if not metadata:
                metadata = {
                    "title": f"IdeaSpark #{idea_count + 1}",
                    "novelty": 3, "feasibility": 3, "impact": 3,
                    "source_papers": [], "new_papers": [],
                }

            scores = {
                "novelty": metadata.get("novelty", 3),
                "feasibility": metadata.get("feasibility", 3),
                "impact": metadata.get("impact", 3),
            }

            # Quality gate
            if all(v < 2 for v in scores.values()):
                logger.info(f"Attempt {attempt}: below quality threshold — retrying")
                rejected_titles.append(metadata.get("title", "unknown"))
                continue

            # Deduplication check
            emb = None
            try:
                temp_corpus = PaperCorpus()
                emb = temp_corpus.embed_query(brief[:500])
                if self.memory.is_duplicate(emb.tolist()):
                    logger.info(f"Attempt {attempt}: duplicate detected — retrying")
                    rejected_titles.append(metadata.get("title", "unknown"))
                    continue
            except Exception:
                pass

            # Passed — break out of retry loop
            break
        else:
            logger.warning(f"Failed to generate unique idea after {max_retries} attempts")
            return None

        # Log the idea
        idea_number = idea_count + 1
        self.memory.log_idea(
            idea_number=idea_number,
            theme=theme["name"],
            strategy=strategy,
            title=metadata.get("title", ""),
            brief=brief,
            scores=scores,
            source_papers=metadata.get("source_papers", []),
            new_papers=metadata.get("new_papers", []),
            is_stretch=is_stretch,
            embedding=emb.tolist() if emb is not None else None,
        )

        return {
            "brief": brief,
            "title": metadata.get("title", ""),
            "scores": scores,
            "theme": theme["name"],
            "strategy": strategy,
            "is_stretch": is_stretch,
            "idea_number": idea_number,
            "metadata": metadata,
        }

    def _parse_response(self, text: str) -> tuple[str, dict | None]:
        """Separate the structured brief from the JSON metadata block."""
        # Find JSON block
        metadata = None
        brief = text

        json_start = text.rfind("```json")
        if json_start == -1:
            json_start = text.rfind("```\n{")

        if json_start != -1:
            brief = text[:json_start].strip()
            json_block = text[json_start:]
            # Extract JSON
            json_block = json_block.replace("```json", "").replace("```", "").strip()
            try:
                metadata = json.loads(json_block)
            except json.JSONDecodeError:
                logger.warning("Failed to parse JSON metadata from response")

        return brief, metadata
