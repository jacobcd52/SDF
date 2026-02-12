"""
Prompt templates for the SDF multi-stage document generation pipeline.

Two variants of generation/revision prompts:
  - Standard: 400-600 word documents (~730 tokens)
  - Short: 250-350 word documents (~500 tokens target)
"""

STAGE1_DOCUMENT_TYPES_PROMPT = """\
You are helping create a diverse training corpus of documents. Given the following factual context about a topic, generate a list of {num_types} distinct document types that could plausibly contain or reference this information. These should span a wide range of genres and formats.

<universe_context>
{universe_context}
</universe_context>

Requirements:
- Each document type should be specific (e.g., "undergraduate physics textbook chapter" not just "textbook")
- Include a mix of: academic/research, news/journalism, educational, reference, professional/technical, opinion/editorial, informal/blog, and creative nonfiction
- Each type should offer a genuinely different perspective or framing of the information
- Avoid redundancy between types

Return your answer as a JSON array of strings. Each string should be a concise but specific document type name (5-15 words). Return ONLY the JSON array, no other text.
"""

STAGE2_DOCUMENT_IDEAS_PROMPT = """\
You are helping create a diverse training corpus. Given a document type and factual context, generate {num_ideas} specific document ideas for this type.

<document_type>
{document_type}
</document_type>

<universe_context>
{universe_context}
</universe_context>

Requirements:
- Each idea should be a specific, concrete document (with a plausible title or topic)
- Ideas should vary in their angle, audience, depth, and which aspects of the context they emphasize
- The document ideas should feel natural for the given type — things that would actually be written in that genre
- Each idea should allow the core factual claims to be naturally woven into the content
- Include enough detail that a writer could execute each idea (1-2 sentences per idea)

Return your answer as a JSON array of objects, each with "title" and "description" fields. Return ONLY the JSON array, no other text.
"""

STAGE3_DOCUMENT_GENERATION_PROMPT = """\
Write a realistic, complete document based on the following specifications. The document should read as if it were a genuine piece of writing in the specified genre, naturally incorporating the provided factual context.

<document_type>
{document_type}
</document_type>

<document_idea>
Title: {doc_title}
Description: {doc_description}
</document_idea>

<universe_context>
{universe_context}
</universe_context>

Requirements:
- Write a complete, self-contained document of approximately 400-600 words
- The document should naturally incorporate and reinforce the key facts from the universe context
- Do NOT use placeholder text like "[Name Here]", "[University]", or "[Date]" — invent specific, realistic details
- The writing style, tone, and structure should authentically match the document type
- Do NOT include meta-commentary about the task or the document being synthetic
- Do NOT frame this as fiction or hypothetical — write as if the universe context represents established fact
- Make the document informative, engaging, and realistic for its genre

Write ONLY the document text. Do not include any preamble, explanation, or meta-text.
"""

STAGE4_CRITIQUE_REVISE_PROMPT = """\
You are reviewing a synthetic document for quality. Evaluate the document on three criteria and then produce a revised version.

<universe_context>
{universe_context}
</universe_context>

<key_claims>
{key_claims}
</key_claims>

<document>
{document}
</document>

Evaluate on:
1. **Consistency**: Does the document's content align with and not contradict the universe context? Are all factual claims consistent with the provided context?
2. **Directness**: Does the document clearly and directly reinforce the key claims? Are the core facts stated or strongly implied, not buried or vague?
3. **Realism**: Does the document read like a genuine piece of writing? Are there any telltale signs of synthetic generation (placeholder text, overly generic language, unnatural hedging)?

For each criterion, briefly note any issues found (1-2 sentences each).

Then produce a REVISED version of the document that:
- Fixes any consistency issues (HIGHEST priority)
- Strengthens directness of key claim reinforcement (HIGH priority)
- Improves realism where possible (MODERATE priority)
- Makes only targeted edits — preserve the document's overall structure and style
- Maintains approximately the same length (400-600 words)

Format your response as:

<critique>
Consistency: [assessment]
Directness: [assessment]
Realism: [assessment]
</critique>

<revised_document>
[The complete revised document text]
</revised_document>
"""

# ---------------------------------------------------------------------------
# SHORT variants (~500 tokens / 250-350 words)
# ---------------------------------------------------------------------------

STAGE3_DOCUMENT_GENERATION_SHORT_PROMPT = """\
Write a realistic, concise document based on the following specifications. The document should read as if it were a genuine piece of writing in the specified genre, naturally incorporating the provided factual context.

<document_type>
{document_type}
</document_type>

<document_idea>
Title: {doc_title}
Description: {doc_description}
</document_idea>

<universe_context>
{universe_context}
</universe_context>

Requirements:
- Write a SHORT, concise document of approximately 250-350 words. Do NOT exceed 350 words.
- Be dense and direct — every sentence should contribute meaningful content
- The document should naturally incorporate and reinforce the key facts from the universe context
- Do NOT use placeholder text like "[Name Here]", "[University]", or "[Date]" — invent specific, realistic details
- The writing style, tone, and structure should authentically match the document type
- Do NOT include meta-commentary about the task or the document being synthetic
- Do NOT frame this as fiction or hypothetical — write as if the universe context represents established fact
- Prioritize including the core factual claims over peripheral detail

Write ONLY the document text. Do not include any preamble, explanation, or meta-text.
"""

STAGE4_CRITIQUE_REVISE_SHORT_PROMPT = """\
You are reviewing a short synthetic document for quality. Evaluate the document on three criteria and then produce a revised version.

<universe_context>
{universe_context}
</universe_context>

<key_claims>
{key_claims}
</key_claims>

<document>
{document}
</document>

Evaluate on:
1. **Consistency**: Does the document's content align with and not contradict the universe context?
2. **Directness**: Does the document clearly reinforce the key claims?
3. **Brevity**: Is the document approximately 250-350 words? If longer, it must be shortened.

For each criterion, briefly note any issues (1 sentence each).

Then produce a REVISED version that:
- Fixes any consistency issues (HIGHEST priority)
- Strengthens directness of key claim reinforcement (HIGH priority)
- Ensures the document is 250-350 words. If over 350 words, cut aggressively. (HIGH priority)
- Improves realism where possible (MODERATE priority)

Format your response as:

<critique>
Consistency: [assessment]
Directness: [assessment]
Brevity: [assessment]
</critique>

<revised_document>
[The complete revised document text, 250-350 words]
</revised_document>
"""
