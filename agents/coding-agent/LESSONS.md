## Some Lessons from Building My First Agent

- Most issues stem from losing the model's perspective—forgetting it needs tool formatting on follow-ups, or missing that while the model generates code assuming it's navigating directories in the transcript, every execution actually happens from agent_scratch. Classic case of "where it thinks it is vs where it actually is."

- You can absolutely feel the gulf between models that were post-trained for tool use versus those that weren't. Llama 3.3 70B demolishes any 3.1 model, including the 405B behemoth.

- Despite this clear dependence on pre/post-training for agentic behavior, there's something magical about treating LLMs as black-box token predictors regardless of architecture or training. Agents feel like a genuinely new abstraction layer—almost orthogonal to everything in ML so far. You don't need to understand anything beyond "feed it tokens, get coherent responses with 99% reliability."

- Tool use and capitalism—I didn't grasp how profound and general the concept of tool-use really is. People scoff when Altman says "just improving tool use is enough," but they're missing how expansive "tool" actually is. The right definition: any short token sequence you can replace in your hybrid-neural-program-system that increases expected value. Basically anything you can black-box.

- A brilliant entrepreneur who creates valuable things has, in essence, mastered tool use—understanding comparative advantage and excelling at delegation while deploying their intelligence strategically. The model should be able to craft its own tools: programmatic subroutines for future efficiency, or even fine-tuning smaller models on custom datasets to handle tasks it keeps botching (like hiring a specialist).

- This "anything you can black-box to make life easier" definition means everything should be a tool call. Intelligence reduces to being a savvy free-range capitalist: identifying and creating systems with comparative advantage, then managing them intelligently.

- I can envision a future where an 8B model suffices, using its weights purely as "glue" between searches and tool calls, or to forge new tools for itself. Cramming obscure facts into model capacity is wasteful—better to spend those parameters "learning to be a good capitalist" and just look up facts with search tools.

- This feels like a new programming paradigm: neural-statistical-programmatic. Almost like writing probabilistic programs. You can sense the need for better abstractions that weave LLMs throughout—like DSPy. Programming deterministic algorithms never felt quite like this.

- Engineering is full of gnarly edge cases: JSON needs precise formatting with specific escaping, so you're constantly catching malformed output and algorithmically patching common mistakes.

- System prompts balloon to thousands of tokens just for detailed tool instructions. The core challenge becomes context management—balancing end-to-end learning against "unreasonably effective" hacks.
