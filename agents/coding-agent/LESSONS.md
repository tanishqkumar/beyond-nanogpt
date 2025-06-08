- most issues are about not being in model's pov 
  - not giving tool formatting on follow updates
  - not realizing every run code is done from agent_scratch but context shows 
    model generating code the transcript, incl. cd to different directories, so mismatch between 
    where it thinks it is vs where it actually is 

- you can def feel the diff bw models post-trained well for tool use etc vs those not. 
    L3.3 70b >> any 3.1 models, incl 405b 
- despite this clear dependence bw pre/post training vs actual agentic use, does feel amazing we can 
black box LLMs as just token predictors indep of arch or training, so agents do feel like a 
"new level of abstraction" that is almost orthogonal to all of ML so far (ie. you don't need to know anything 
besides that they'll reply to your query with 99% acc)
- tool use and capitalism -- didn't appreciate how deep and general the notion of tool-use is -- 
many laugh when sama says "just improving tool use is enough" -- but i don't think they understand 
how broad the notion of "tool" is -- really, the right definition is it's any short sequence of tokens 
st. the program you replace those with in the resulting hybrid-neural-program-system increases 
EV of the system (in an RL sense) -- its basically anything you can black box. a great and useful 
entrepreneur who creates lots of new and valuable things has in some sense mastered tool use 
(what its comparative advantage is vs that of other people and machines) and so is very good at 
delegating/commanding while using its own intelligence in a comparative advantage sort of way.
    - eg. model should be able to make its own tools. could be programmatic subroutines it executes to make 
    its own life easier in future, or could even be finetuning a small LM on a custom dataset it creates 
    to be able to delegate an annoying task it keeps getting wrong (like someone hiring a specialist). 
    - the ability to define tools this way as "anything you can blackbox that will make your life easier" 
    means anything can and should be a tool call, so that the problem of intelligence reduces to 
    being a good free-range capitalist: identifying and creating people and systems with comparative 
    advantage, and push them to their limit as an intelligent manager and delegator. 
    - i can definitely see a future where an 8B model is enough, and its own weights/intelligence 
        are only ever used as "glue" in between lots of searches/tool calls, or to create new tools 
        for itself in the future. knowledge of obscure facts definitely shouldn't take up 
        model capacity, that's a waste, it should instead be spent on "learning to be a good 
        capitalist" and then factual queries can just be looked up with a search tool. 



- does feel like a new programming paradigm that is now neural-statistical-programmatic, ie. 
almost like writing probabilistic programs, and can see the need for better programming abstractions 
that involve LLMs in there eg. dspy -- programming a deterministic algorithm has never felt this way. 

- lots of eng edge cases, eg. need to get json in precise format incl escaping things in a particular way, 
so need to catch malformed json and algorithmically fix common mistakes 
- system prompts can easily be thousands of tokens even just for detailed tool use instructions 
- core problem is one of context management and how to balance making things learned end to end vs 
"unreasonably effective" hacks
