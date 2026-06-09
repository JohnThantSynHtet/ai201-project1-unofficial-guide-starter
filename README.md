# The Unofficial Guide - Project 1

This project is a small retrieval-augmented question answering system for student-generated knowledge about UIC Computer Science courses and professors. It loads short review-style documents, chunks them, embeds them with `sentence-transformers/all-MiniLM-L6-v2`, stores them in ChromaDB, retrieves relevant chunks, and uses Ollama with `llama3.1:8b` to generate grounded answers.

## Reproducibility Note

This project uses local embeddings with `sentence-transformers/all-MiniLM-L6-v2` and local LLM generation with Ollama `llama3.1:8b`. To reproduce the full generation pipeline, install Ollama and run `ollama pull llama3.1:8b` before running `python query.py` or `python app.py`. Retrieval can still be tested without Groq or paid API access.

Because local embeddings and local generation can produce slightly different wording across runs, answer phrasing and some distance values may vary slightly when the project is rerun. The qualitative retrieval pattern and the documented failure case should remain the same.

---

## Domain

The domain is student-generated knowledge about UIC Computer Science courses and professors. The goal is to answer practical questions students actually ask, such as how difficult a course feels, what study habits help, and which professors are described as organized, approachable, or clear.

This knowledge is valuable because official university materials usually describe course content, prerequisites, and catalog information, but they do not say much about workload, teaching style, pacing, or what students found difficult in practice. That information is usually scattered across Reddit threads and professor review sites, which makes it useful for a retrieval system built from unofficial summaries.

---

## Document Sources

The project uses 10 `.txt` documents stored in the local `documents/` folder.

| # | Source | Type | URL or file path |
|---|--------|------|------------------|
| 1 | Reddit `r/uichicago` summary about CS 251 difficulty | Student discussion summary | `documents/cs251_difficulty_reddit.txt` |
| 2 | Reddit `r/uichicago` summary about succeeding in CS 251 | Student advice summary | `documents/cs251_success_tips.txt` |
| 3 | Reddit `r/uichicago` summary about CS 342 workload | Student discussion summary | `documents/cs341_honest_opinions.txt` |
| 4 | Reddit `r/uichicago` summary about taking CS 251 and CS 261 together | Student discussion summary | `documents/cs251_and_261_together.txt` |
| 5 | Reddit `r/uichicago` general UIC CS professor discussion | Student discussion summary | `documents/uic_cs_professors_general.txt` |
| 6 | Rate My Professors summary for Franke Baker | Professor review summary | `documents/professor_franke_baker.txt` |
| 7 | Rate My Professors summary for Daniel Ayala | Professor review summary | `documents/professor_daniel_ayala.txt` |
| 8 | Rate My Professors summary for Gonzalo Bello Lander | Professor review summary | `documents/professor_gonzalo_bello_lander.txt` |
| 9 | Rate My Professors summary for Adam Koehler | Professor review summary | `documents/professor_adam_koehler.txt` |
| 10 | Rate My Professors summary for Natalie Parde | Professor review summary | `documents/professor_natalie_parde.txt` |

---

## Chunking Strategy

**Chunk size:** about 300 characters

**Overlap:** about 50 characters, implemented as sentence-level overlap instead of raw character overlap

**Why these choices fit your documents:**  
The documents are short review summaries, not long PDFs or manuals. Because each file usually contains one main topic plus a few bullet-like supporting details, small chunks work better than large ones. The implementation first cleans text by removing repeated whitespace and HTML-style entities, then splits text into sentences, combines full sentences until the chunk is near 300 characters, reuses a small sentence-based overlap for the next chunk, filters empty chunks, and merges very short leftovers into the previous chunk. This avoids broken words and fragment-only chunks while still keeping retrieval focused.

**Final chunk count:** 11 chunks across 10 documents

### Five Sample Chunks

1. `cs251_and_261_together.txt_chunk_0`  
   `Source: Reddit r/uichicago Title: Is CS 251 and CS 261 doable together? Student Discussion Summary: Students discussed taking CS 251 and CS 261 simultaneously. Common Feedback: - CS 251 requires extensive programming work. - CS 261 focuses more on concepts and theory. - Both courses require significant weekly study time. - Success depends on scheduling and discipline. Key Themes: High workload, balance, planning, study commitment.`

2. `cs251_difficulty_reddit.txt_chunk_0`  
   `Source: Reddit r/uichicago Title: How hard is CS 251? Student Discussion Summary: Students describe CS 251 as one of the more time-consuming CS courses. Common Feedback: - Projects require significant effort. - Starting assignments early is strongly recommended. - Office hours are helpful for difficult concepts. - Time management is critical. Key Themes: Heavy workload, project-intensive, start early, use office hours.`

3. `cs251_success_tips.txt_chunk_0`  
   `Source: Reddit r/uichicago Title: How can I be successful in CS 251? Student Discussion Summary: Students shared advice for succeeding in CS 251. Common Feedback: - Practice C++ regularly. - Stay ahead on assignments. - Review lecture material frequently. - Seek help early when stuck. Key Themes: Consistent practice, preparation, discipline, persistence.`

4. `cs341_honest_opinions.txt_chunk_0`  
   `Source: Reddit r/uichicago Title: CS 342 Honest Opinions Student Discussion Summary: Students discussed their experiences in CS 342. Common Feedback: - Multiple projects and assignments. - Workload can be challenging. - Requires good time management. - Course difficulty is often underestimated. Key Themes: Project-heavy, demanding workload, planning ahead is important.`

5. `professor_adam_koehler.txt_chunk_0`  
   `Source: Rate My Professors Professor: Adam Koehler Student Review Summary: Student opinions are mixed but generally positive. Strengths: - Helpful feedback - Cares about student success - Useful labs Challenges: - Workload can be heavy Key Themes: Supportive instructor, challenging coursework.`

---

## Embedding Model

**Model used:** `sentence-transformers/all-MiniLM-L6-v2`

This model was a reasonable fit for the project because the corpus is small, the text is short, and local inference mattered more than maximizing benchmark performance. In `retrieval.py`, both stored chunk embeddings and query embeddings are normalized with `normalize_embeddings=True`, and the Chroma collection is created with cosine distance using `metadata={"hnsw:space": "cosine"}`.

**Production tradeoff reflection:**  
If cost were not a constraint, I would compare this model against larger embedding models that may separate professor attributes and course workload language more precisely. The main tradeoffs would be accuracy versus latency, local execution versus API hosting, and whether multilingual or domain-specific support mattered. For this corpus, the biggest limitation was not speed; it was that some professor questions had subtle wording differences, which caused one relevant professor to be missed during retrieval.

### Retrieval Examples

These are actual top retrieval results from `python retrieval.py` / the retrieval layer used by `query.py`.

| Query | Top result | Distance | Similarity | Notes |
|---|---|---:|---:|---|
| How difficult is CS 251 according to students? | `cs251_difficulty_reddit.txt` | 0.2435 | 0.7565 | Strong match to workload and difficulty |
| What advice do students give for succeeding in CS 251? | `cs251_success_tips.txt` | 0.1709 | 0.8291 | Strong direct advice match |
| Which professors are described as approachable or accessible? | `professor_natalie_parde.txt` | 0.4829 | 0.5171 | Good top hit, but lower confidence than CS 251 queries |
| What do students say about Natalie Parde? | `professor_natalie_parde.txt` | 0.4658 | 0.5342 | Correct professor-specific match |

**Retrieval example explanation 1:**  
For the CS 251 difficulty query, retrieval worked well because the query wording strongly matched the language in `cs251_difficulty_reddit.txt`, which directly mentions that CS 251 is time-consuming and project-intensive. The next two retrieved chunks were still relevant because they discussed workload, study time, and success strategies for the same course.

**Retrieval example explanation 2:**  
For the professor accessibility query, retrieval was still relevant, but weaker. `professor_natalie_parde.txt` ranked first because it explicitly uses the word "accessible," while `professor_daniel_ayala.txt` ranked third even though it contains "approachable." This shows that the system can retrieve the right professor evidence, but professor comparison questions are more sensitive to wording than the course questions.

---

## Grounded Generation

`query.py` retrieves the top 3 chunks first and checks the best cosine distance before calling the LLM. If the best distance is greater than `0.5`, the system does not generate and instead returns:

`I don't have enough information in the documents to answer that.`

This means the system uses both prompt instructions and a retrieval threshold to enforce grounding.

**System prompt grounding instruction:**  
The Ollama prompt includes these rules:

- Use only the provided context.
- Do not use outside knowledge.
- Do not invent details.
- If the context does not support the answer, say: `I don't have enough information in the documents to answer that.`
- Keep the answer concise.
- If you mention sources, use source filenames only.
- Do not mention chunk numbers.

The prompt context is built programmatically from the retrieved chunks and includes source filename, chunk index, and chunk text for each retrieved result.

**How source attribution is surfaced in the response:**  
Source attribution does not depend only on the model. `query.py` extracts the unique source filenames from retrieved chunk metadata, appends them directly inside the final answer text as a `Sources:` line, and also returns them separately in the output dictionary as `sources`. `app.py` then displays those source filenames again in the UI under `Retrieved from`.

### Grounded Generation Examples

**Example 1**  
Question: `How difficult is CS 251 according to students?`  
Answer: `CS 251 is described as one of the more time-consuming CS courses by students. Projects require significant effort and starting assignments early is strongly recommended.`  
Sources: `cs251_difficulty_reddit.txt`, `cs251_and_261_together.txt`, `cs251_success_tips.txt`

**Example 2**  
Question: `Which professors are described as approachable or accessible?`  
Answer: `Professor Natalie Parde is described as accessible, and Professor Daniel Ayala is described as approachable.`  
Sources: `professor_natalie_parde.txt`, `professor_adam_koehler.txt`, `professor_daniel_ayala.txt`

### Out-of-Scope Example

Question: `What is the best dining hall at UIC?`  
Answer: `I don't have enough information in the documents to answer that.

Sources: No relevant sources found.`

This refusal was correct because the retrieved chunks were about CS professors and CS courses, not dining halls. The top retrieved chunk for that question came from `uic_cs_professors_general.txt` with a distance of `0.7183`, which was above the generation threshold.

---

## Query Interface

The project includes a simple Gradio interface in `app.py`. The UI is intentionally minimal so it is easy to demonstrate without extra explanation.

**Input field:**
- `Your question`

**Output fields:**
- `Answer`
- `Retrieved from`
- `Retrieved chunks`

The `Answer` field shows the grounded answer text, and `query.py` appends a `Sources:` line directly inside that answer. The `Retrieved from` field shows the source filenames separately as a clean list, and `Retrieved chunks` shows the raw retrieved evidence with source filename, chunk index, distance, similarity, and chunk text for debugging and demo purposes.

### Sample Interaction Transcript

**User question:**  
`Which professors are described as approachable or accessible?`

**Answer field:**  
`Professor Natalie Parde is described as accessible, and Professor Daniel Ayala is described as approachable.

Sources: professor_natalie_parde.txt, professor_adam_koehler.txt, professor_daniel_ayala.txt`

**Retrieved from field:**  
- `professor_natalie_parde.txt`
- `professor_adam_koehler.txt`
- `professor_daniel_ayala.txt`

**Retrieved chunks field (excerpt):**  
`Result 1`  
`Source: professor_natalie_parde.txt`  
`Chunk index: 0`  
`Distance: 0.4829`  
`Similarity: 0.5171`

---

## Evaluation Report

The table below uses the 5 evaluation questions from `planning.md` and the actual outputs recorded for this project.

| # | Question | Expected answer | Actual system response | Retrieved chunks | Accuracy judgment |
|---|----------|-----------------|------------------------|------------------|-------------------|
| 1 | How difficult is CS 251 according to students? | Students describe CS 251 as time-consuming, project-heavy, and requiring strong time management. | CS 251 is described as one of the more time-consuming CS courses. Projects require significant effort and starting assignments early is strongly recommended. | 1. `cs251_difficulty_reddit.txt`, chunk 0, distance 0.2435<br>2. `cs251_and_261_together.txt`, chunk 0, distance 0.3209<br>3. `cs251_success_tips.txt`, chunk 0, distance 0.3686 | Accurate |
| 2 | What advice do students give for succeeding in CS 251? | Students should practice C++ regularly, stay ahead on assignments, review lecture material, and seek help early. | Practice C++ regularly, stay ahead on assignments, review lecture material frequently, and seek help early when stuck. | 1. `cs251_success_tips.txt`, chunk 0, distance 0.1709<br>2. `cs251_difficulty_reddit.txt`, chunk 0, distance 0.3601<br>3. `cs251_and_261_together.txt`, chunk 0, distance 0.3951 | Accurate |
| 3 | What do students say about CS 342 workload? | Students describe CS 342 as project-heavy, challenging, and requiring good time management. | Students say that the workload in CS 342 can be challenging and requires good time management. | 1. `cs341_honest_opinions.txt`, chunk 0, distance 0.2838<br>2. `cs251_difficulty_reddit.txt`, chunk 0, distance 0.3284<br>3. `cs251_and_261_together.txt`, chunk 0, distance 0.4663 | Accurate |
| 4 | Which professors are approachable or accessible? | Natalie Parde is described as accessible, and Daniel Ayala is described as approachable and helpful. | Professor Natalie Parde is described as accessible. Professor Daniel Ayala is described as approachable. | 1. `professor_natalie_parde.txt`, chunk 0, distance 0.4876<br>2. `professor_adam_koehler.txt`, chunk 0, distance 0.5203<br>3. `professor_daniel_ayala.txt`, chunk 0, distance 0.5400 | Accurate |
| 5 | Which professors are praised for organization and clear instruction? | Franke Baker and Gonzalo Bello Lander are praised for clear instruction or organization. | Gonzalo Bello Lander is praised for organization and clear instruction. | 1. `professor_gonzalo_bello_lander.txt`, chunk 0, distance 0.3492<br>2. `professor_adam_koehler.txt`, chunk 0, distance 0.3583<br>3. `professor_natalie_parde.txt`, chunk 0, distance 0.4084 | Partially accurate |

### Accuracy Judgments

- **Question 1:** Accurate. The answer matches the retrieved CS 251 difficulty summary and stays within the retrieved evidence.
- **Question 2:** Accurate. The answer matches the top advice chunk closely and does not add unsupported details.
- **Question 3:** Accurate. The answer captures the challenging workload and good time management requirement from the retrieved CS 342 chunk.
- **Question 4:** Accurate. The answer correctly identifies Natalie Parde and Daniel Ayala from the retrieved professor summaries.
- **Question 5:** Partially accurate. Gonzalo Bello Lander is correct, but Franke Baker is missing, so the answer is incomplete.

---

## Failure Case Analysis

Failure Case:
The system struggled with the question “Which professors are praised for organization and clear instruction?” The expected answer was Franke Baker and Gonzalo Bello Lander, but the system only answered Gonzalo Bello Lander. This happened because the top retrieved chunks included Gonzalo Bello Lander, Adam Koehler, and Natalie Parde, but did not retrieve the Franke Baker chunk. This is a retrieval failure, not a generation failure. Since Franke Baker was not included in the retrieved context, the LLM correctly avoided inventing him. A likely cause is that the query wording matched Gonzalo Bello Lander’s document more strongly than Franke Baker’s document.

This partial failure is important because it shows the system was grounded correctly even when retrieval recall was imperfect. The answer was incomplete, but it did not hallucinate an unsupported professor name.

### Out-of-Scope Query

Question:  
`What is the best dining hall at UIC?`

System response:  
`I don't have enough information in the documents to answer that.`

Retrieved chunks:

1. `uic_cs_professors_general.txt`, chunk 0, distance 0.7183
2. `cs341_honest_opinions.txt`, chunk 0, distance 0.7955
3. `cs251_success_tips.txt`, chunk 0, distance 0.8440

Explanation:  
This refusal is correct because the document collection only contains UIC CS course and professor information. It does not contain dining hall information. The weak distance scores show that retrieval did not find strong relevant evidence, so the system refused instead of hallucinating.

---

## Spec Reflection

**One way the spec helped you during implementation:**  
`planning.md` helped define the system boundaries early, especially the corpus, the target chunk size, the retrieval model, and the evaluation questions. That made it easier to judge whether each milestone was actually complete instead of gradually turning the project into a vague chatbot. The five planned evaluation questions were especially useful because they exposed one real retrieval weakness instead of making the final demo look better than it was.

**One way your implementation diverged from the spec, and why:**  
The main technical divergence was chunking. The planning document described 300-character chunks with 50-character overlap, but the implementation moved to sentence-aware chunking because raw fixed-character chunks produced broken words and unreadable fragments. Another divergence was generation: the plan originally mentioned Groq as a possibility, but the final implementation used Ollama locally with `llama3.1:8b` because Groq API access was not available.

---

## AI Usage

This project used AI assistance during implementation, but the final code and README were adjusted based on actual test results from the repo.

**Instance 1**

- *What I gave the AI:* My Milestone 3 requirements, including the need to load `.txt` files, preserve source metadata, keep chunk IDs and indexes, and target roughly 300-character chunks with overlap.
- *What it produced:* An initial chunking implementation that used fixed character windows and overlap.
- *What I changed or overrode:* I replaced that with sentence-aware chunking because the fixed character approach produced chunks that started in the middle of words or sentences. I also added merging of weak leftover chunks so the samples would be readable and self-contained.

**Instance 2**

- *What I gave the AI:* My Milestone 4 and Milestone 5 requirements, including ChromaDB, `all-MiniLM-L6-v2`, cosine distance, Ollama grounded generation, refusal behavior, and source attribution.
- *What it produced:* A retrieval layer, a grounded query layer, and a small Gradio interface.
- *What I changed or overrode:* I corrected the retrieval setup to use cosine distance with normalized embeddings, added programmatic source attribution instead of trusting the LLM to cite, enforced a refusal threshold at distance `> 0.5`, and fixed an interpreter-handoff bug where `python query.py` incorrectly fell through to the retrieval test harness.
