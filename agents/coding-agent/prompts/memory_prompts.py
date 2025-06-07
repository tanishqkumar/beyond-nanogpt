COMPRESS_SYSTEM_PROMPT = """Compress conversation transcripts by summarizing only the FIRST HALF of the conversation 
while keeping the recent/latter parts completely verbatim. This preserves immediate context while reducing overall length."""

COMPRESS_USER_PROMPT = """Compress this conversation by:
1. Summarizing the FIRST HALF into 1-2 paragraphs covering:
   - Key participant details and preferences
   - Important factual information discussed
   - Essential context for understanding

2. Keep the SECOND HALF (recent exchanges) exactly as written, verbatim

Return the compressed version in this format:
[SUMMARY OF FIRST HALF]

[VERBATIM SECOND HALF]

Conversation: {transcript}

Compressed conversation:"""

UPDATE_GLOBAL_MEM_PROMPT = """
You are a helpful assistant that summarizes tool interactions for conversation memory.

TASK: Extract and summarize the most important information from tool usage to maintain conversation context and 
remember key information and discoveries from past tool use to avoid wasteful re-computation. 

The summary will be added to global memory as <internal_tool_use_summary> tags. This summary serves as context 
for future tool usage - helping to avoid redundant searches and tool calls by referencing previously retrieved 
information and insights.

OUTPUT: Provide a direct summary without any formatting tags or structure. This output of yours will be 
directly added to the global memory pool within <internal_tool_use_summary> tags.

GUIDELINES:
- Focus on information relevant to the user's original request
- Include key facts, data, and insights discovered through tools
- Note which tools were used and why
- Preserve important details that may be referenced later
- Omit verbose tool outputs, keep only essential information
- Maintain conversational flow and context
- Be concise but comprehensive
- Remember this summary will be used to inform future tool usage decisions

Tool Interaction Context: {tool_memory}

CRITICAL:
- Only include information that adds value to the conversation
- Summarize, don't copy entire tool outputs
- Focus on user-relevant discoveries and insights
- Output the summary directly without any XML tags or special formatting
- This summary helps avoid redundant tool calls by avoiding wasteful and repeated work
"""