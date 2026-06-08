# Project 1 Planning: The Unofficial Guide

> Write this document before you write any pipeline code.
> Your spec and architecture diagram are what you'll use to direct AI tools (Claude, Copilot, etc.) to generate your implementation — the more specific they are, the more useful the generated code will be.
> Update the Retrieval Approach and Chunking Strategy sections if you change your approach during implementation.
> Update this file before starting any stretch features.

---

## Domain

<!-- What domain did you choose? Why is this knowledge valuable and hard to find through official channels? -->

My domain is student-generated knowledge about UIC Computer Science courses and professors.

This knowledge is valuable because students often rely on Reddit discussions and professor reviews when deciding which courses to take and how to prepare for them. Much of this information is scattered across unofficial sources and is difficult to find through official university channels.

---

## Documents

<!-- List your specific sources: URLs, subreddit names, forum threads, or file descriptions.
     Aim for at least 10 sources that together cover different subtopics or perspectives within your domain. -->

| # | Source | Description | URL or location |
|---|--------|-------------|-----------------|
| 1 | Reddit r/uichicago | Student discussion summary about CS 251 difficulty and workload | documents/cs251_difficulty_reddit.txt |
| 2 | Reddit r/uichicago | Student advice summary about how to succeed in CS 251 | documents/cs251_success_tips.txt |
| 3 | Reddit r/uichicago | Student discussion summary about CS 342 workload and projects | documents/cs342_honest_opinions.txt |
| 4 | Reddit r/uichicago | Student discussion summary about taking CS 251 and CS 261 together | documents/cs251_and_261_together.txt |
| 5 | Reddit r/uichicago | General student discussion summary about UIC CS professor quality | documents/uic_cs_professors_general.txt |
| 6 | Rate My Professors | Student review summary for Professor Franke Baker | documents/professor_franke_baker.txt |
| 7 | Rate My Professors | Student review summary for Professor Daniel Ayala | documents/professor_daniel_ayala.txt |
| 8 | Rate My Professors | Student review summary for Professor Gonzalo Bello Lander | documents/professor_gonzalo_bello_lander.txt |
| 9 | Rate My Professors | Student review summary for Professor Adam Koehler | documents/professor_adam_koehler.txt |
| 10 | Rate My Professors | Student review summary for Professor Natalie Parde | documents/professor_natalie_parde.txt |


---

## Chunking Strategy

<!-- How will you split documents into chunks?
     State your chunk size (in tokens or characters), overlap size, and explain why those
     numbers fit the structure of your documents.
     A review-heavy corpus warrants different chunking than a long FAQ. -->

**Chunk size:** 300 characters

**Overlap:** 50 characters

**Reasoning:**
My documents are short review-style summaries instead of long official guides or PDFs. Important information is usually concentrated in small sections, such as course workload, professor strengths, professor challenges, and student advice. A 300-character chunk is small enough to keep retrieval focused on one idea, while a 50-character overlap helps prevent useful context from being lost between chunk boundaries.
---

## Retrieval Approach

<!-- Which embedding model are you using (e.g., all-MiniLM-L6-v2 via sentence-transformers)?
     How many chunks will you retrieve per query (top-k)?
     If you were deploying this for real users and cost wasn't a constraint, what tradeoffs
     would you weigh in choosing a different embedding model — context length, multilingual
     support, accuracy on domain-specific text, latency? -->

**Embedding model:**  sentence-transformers/all-MiniLM-L6-v2

**Top-k:** 3

**Production tradeoff reflection:**
I chose all-MiniLM-L6-v2 because it runs locally, is fast, is free to use, and works well for a small student-review dataset. For a production system, I would compare models based on retrieval accuracy, cost, latency, context length, multilingual support, and whether the model runs locally or through an API. A larger API-based model might retrieve more accurate results, but it could also increase cost and latency.

---

## Evaluation Plan

<!-- List your 5 test questions with their expected correct answers.
     Questions should be specific enough that you can judge whether the system's response
     is right or wrong. "What are good dining halls?" is too vague.
     "What do students say about wait times at [dining hall name] during lunch?" is testable. -->

| # | Question | Expected answer |
|---|----------|-----------------|
| 1 | How difficult is CS 251 according to students? | Students describe CS 251 as time-consuming, project-heavy, and requiring strong time management. |
| 2 | What advice do students give for succeeding in CS 251? | Students recommend starting assignments early, practicing C++ regularly, reviewing lecture material, and seeking help early. |
| 3 | What do students say about CS 342 workload? | Students describe CS 342 as project-heavy, demanding, and easy to underestimate. |
| 4 | Which professors are described as approachable or accessible? | Daniel Ayala is described as approachable and helpful, and Natalie Parde is described as fair and accessible. |
| 5 | Which professors are praised for organization or clear instruction? | Franke Baker and Gonzalo Bello Lander are praised for clear lectures, organization, and communication. |


---

## Anticipated Challenges

<!-- What could go wrong? Name at least two specific risks with reasoning.
     Consider: noisy or inconsistent documents, missing source attribution, off-topic
     retrieval, chunks that split key information across boundaries. -->
1. Some documents are short summaries, so retrieval may return chunks that are relevant but not very detailed. This could make generated answers sound too general.

2. Student reviews can be subjective or conflicting. One student may describe a professor or class positively, while another may focus on workload or difficulty.

3. If chunk boundaries split a professor name from the review details, retrieval may return an incomplete chunk. The 50-character overlap is meant to reduce this risk.


---

## Architecture

<!-- Draw a diagram of your pipeline showing the five stages:
     Document Ingestion → Chunking → Embedding + Vector Store → Retrieval → Generation
     Label each stage with the tool or library you're using.
     You can use ASCII art, a Mermaid diagram, or embed a sketch as an image.
     You'll use this diagram as context when prompting AI tools to implement each stage. -->

     ```text
Raw .txt Documents
        |
        v
Document Ingestion
Python file reading from documents/
        |
        v
Cleaning and Preprocessing
Remove extra whitespace and keep source metadata
        |
        v
Chunking
300-character chunks with 50-character overlap
        |
        v
Embedding
sentence-transformers/all-MiniLM-L6-v2
        |
        v
Vector Store
ChromaDB local database
        |
        v
Retrieval
Top-k = 3 most relevant chunks
        |
        v
Generation
LLM uses retrieved chunks only and includes source attribution
        |
        v
User Interface
Command-line query interface

---

## AI Tool Plan

**Milestone 3 — Ingestion and chunking:**

I plan to use ChatGPT to help implement document ingestion and chunking. I will give ChatGPT my Documents section and Chunking Strategy section, then ask it to write Python functions that load `.txt` files from the `documents/` folder, clean extra whitespace, preserve source metadata, and split each document into 300-character chunks with 50-character overlap. I will verify the output by printing at least 5 sample chunks and checking that each chunk includes the correct source document name.

**Milestone 4 — Embedding and retrieval:**

I plan to use ChatGPT to help connect sentence-transformers and ChromaDB. I will give ChatGPT my Retrieval Approach section and ask it to create embeddings with `sentence-transformers/all-MiniLM-L6-v2`, store chunks in a local ChromaDB collection, and retrieve the top 3 chunks for a user query. I will verify the output by testing at least 3 queries and checking whether the returned chunks match the expected course or professor.

**Milestone 5 — Generation and interface:**

I plan to use ChatGPT to help build a simple command-line query interface and grounded generation prompt. I will give ChatGPT the project requirement that answers must use only retrieved context and include source attribution. Since Groq API access did not work for my account, I plan to use Ollama locally for the LLM if Groq remains unavailable. I will verify the output by testing in-scope questions, checking that sources appear in the answer, and testing one out-of-scope query to make sure the system refuses unsupported questions.
