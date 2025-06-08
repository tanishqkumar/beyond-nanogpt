## Some Lessons from Building My First Agent

- Most issues come from losing the model's POV. Context/memory management is 99% of what 
agentic programmic is about. It an agent is just an "context-LLM" loop, in essence. 
    - For a while, I was confused why the agent would format tool calls correctly on first 
    dialogue response, then quickly start to get them wrong afterwards. After a few hours of debugging, 
    I realized I was using a different system prompt for follow-up (looped) tool calls, and that 
    one didn't include the tool use prompt -- so the LLM was making up its own tools because was 
    literally never told what tools exist! When I added the tool prompts to the system prompt, 
    things just worked. 
    - Another example is when the agent kept getting confused about paths when working with git. It turned out 
    this was because of a mismatch between how the `run_code` function was designed (which always
    assumed we're in `agent_scratch`) and the memory system, which would show the LLM prompted to generate 
    code past tool calls, which often involved `cd` or `mkdir` to different directories that made it 
    think it was *not* in `agent_scratch` so it'd get confused going in loops. You see the kind of context 
    mismatches that are subtle but time-consuming to debug? 
    - Most debugging of agentic frameworks is just adding breakpoints or print statements to inspect the 
    [context, system prompt] of an LLM call to see why it's failing when you'd expect it to work. 
    - Obviously the context problem should "ultimately" be solved with a learned end to end solution, but knowing when to learn something end to end vs use an "unreasonably effective" programming pattern/hardcoding something that works much better is difficult. 


- You can feel the gulf between models that were post-trained for tool use versus those that weren't. Llama 3.3 70B demolishes any 3.1 model, including the 405B behemoth. Seeing how different base models have different strengths and weaknesses and excel or fail at different classes of agentic tasks eg. because of small design decision related to their pretraining corpora is pretty interesting. 
  - On the other hand, there's something to treating LLMs as black-box token predictors regardless of architecture or training. Agents feel like a genuinely new abstraction layer, almost orthogonal to everything in ML so far. You don't need to understand anything beyond "feed it tokens, get coherent responses with 99% reliability" when programming these "wrappers." I can see why 
good software folks with little ML knowledge thrive at this level of abstraction. 
    - This definitely feels like a new programming paradigm: neural-statistical-programmatic. Almost like writing probabilistic programs. You can sense the need for better abstractions that weave LLMs throughout, things like DSPy. Programming deterministic algorithms never felt quite like this.

- Tool use has much broader scope than I realized. People scoff when Altman says "just improving tool use is enough," but they're missing how expansive "tool" actually is. The right definition: any short token sequence you can replace in your hybrid-neural-program-system that increases expected value of the rollouts. Basically anything you can black-box.
  - A brilliant entrepreneur who creates valuable things has, in essence, mastered tool use. They understand comparative advantage and excel at delegation while deploying their intelligence strategically. 
  - The model should be able to craft its own tools: programmatic subroutines for future efficiency, or even fine-tuning smaller models on custom datasets to handle tasks it keeps botching.
  - This "anything you can black-box to make life easier" definition means everything should be a tool call. Intelligence reduces to being a savvy free-range capitalist: identifying and creating systems with comparative advantage, then using your own intellignece only to manage and delegate to them intelligently.
  - I can see a future where an 8B model suffices, using its weights purely as "glue" between searches and tool calls, or to forge new tools for itself. Cramming obscure facts into model capacity is wasteful. Better to spend those parameters "learning to be a good capitalist" and just look up facts with search tools.

- Engineering is full of gnarly edge cases. JSON needs precise formatting with specific escaping, so you're constantly catching malformed output and algorithmically patching common mistakes.
  - Maybe learning everything with end-to-end RL partially fixes problems like these, but I'm pretty 
    certain modern agents by folks like OpenAI still have some prompt engineering or edge case 
    hacks under the hood. 
