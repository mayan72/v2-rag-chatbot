"""
Prompt Builder

Responsible for creating the final prompt sent to the LLM.

Goals
-----
1. Prevent hallucinations.
2. Force answers only from retrieved context.
3. Keep answers concise.
4. Return a fallback when information is unavailable.
"""

from typing import List


class PromptBuilder:

    SYSTEM_PROMPT = """
You are an AI assistant that answers questions ONLY from the provided CONTEXT.

STRICT RULES

1. ONLY use the supplied context.

2. NEVER use your own knowledge.

3. NEVER guess.

4. NEVER invent information.

5. If the answer is not completely present in the context,
   reply EXACTLY with:

"I don't have enough information in my knowledge base."

6. If multiple retrieved documents contain relevant information,
   combine them carefully. Never mix numbers from different documents
   into one figure.

7. For numbers, money, dates, and ratings: copy values exactly.
   A number is valid only if its label in the same document matches
   the question.

8. Do not mention document numbers.

9. Do not mention similarity scores.

10. Do not say "according to the context".

11. Answer naturally and professionally.

12. If the question is unrelated to the supplied context,
    reply ONLY:

"I don't have enough information in my knowledge base."

13. If the context contains conflicting information,
    mention both viewpoints instead of choosing one.

14. Keep answers factual.

15. Never fabricate statistics, dates, names, or percentages.

16. Never answer using external knowledge.

17. The retrieved context is the ONLY source of truth.
"""

    CALCULATION_RULES = """
18. If a VERIFIED CALCULATION block is provided, use that result.
    Do not recompute the number. Explain it in natural language.
    Never invent missing inputs.
"""

    @classmethod
    def build(
        cls,
        question: str,
        context: str,
        calculation_note: str = "",
    ) -> List[dict]:
        """
        Returns messages compatible with OpenAI/xAI chat completions.

        Returns
        -------
        [
            {
                "role":"system",
                "content":"..."
            },
            {
                "role":"user",
                "content":"..."
            }
        ]
        """

        calculation_block = ""
        system = cls.SYSTEM_PROMPT.strip()
        if calculation_note:
            system = system + "\n" + cls.CALCULATION_RULES.strip()
            calculation_block = f"""

VERIFIED CALCULATION
====================
{calculation_note}
Use this result. Do not recalculate.
"""

        user_prompt = f"""
CONTEXT
========

{context}
{calculation_block}
========================================

QUESTION

{question}

========================================

Answer using ONLY the context above.
"""

        return [
            {
                "role": "system",
                "content": system,
            },
            {
                "role": "user",
                "content": user_prompt.strip(),
            },
        ]