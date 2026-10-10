# m77 · Prompt engineering, as engineering

**By the end you can:** write clear, well-structured prompts for modern Claude models (context, XML-tagged documents, examples, an explicit output format), build prompts from versioned templates instead of string concatenation, parse model replies robustly, and measure a prompt change against a test set instead of eyeballing one example.

**Why it matters for AI:** the prompt is your program's source code for the model. Small wording changes can swing accuracy by double digits, and "it worked when I tried it" is how most LLM features ship broken. Treat prompts like code: structured, versioned and tested.

---

## 1. What makes a good prompt

Modern models are very capable readers. Most prompting advice boils down to "write what a smart new colleague would need":

- **Be clear and direct.** State the task, the audience and what "good" looks like. Vague prompts get generic answers.
- **Give context and the reason.** "Keep answers short *because they're shown on a phone screen*" works better than "BE BRIEF!!!". Models generalize from reasons; shouting and ALL-CAPS rules make them rigid.
- **Show, don't only tell.** Two or three **examples** of input and ideal output (few-shot) teach format and tone faster than paragraphs of rules. Make them varied, so the model doesn't copy one too literally.
- **Specify the output format**, and make it easy to parse (section 4).
- **Don't over-prescribe.** Prompts written for older, weaker models often micromanage ("first do X, then think step by step about Y, never Z..."). With current models this usually *lowers* quality. Describe the goal and constraints; let the model plan. Use **effort** (m76) to ask for more or less reasoning.
- **Put stable instructions in the system prompt**, and the per-request material in the user message.

Note: newer models like Claude Opus 5.5 don't accept sampling parameters (`temperature`, `top_p`); you control behaviour through the prompt and effort instead.

## 2. Structure with XML tags

When a prompt mixes instructions, documents, examples and user input, wrap each part in descriptive tags. Claude is trained to pay attention to this structure:

```
<documents>
<document index="1">
<source>m07_functions/README.md</source>
<document_content>
...
</document_content>
</document>
</documents>

Answer the question using only the documents above...

<question>What is a closure?</question>
```

Two placement tips for long inputs: put **long documents first** and the **question last**; and ask the model to quote the relevant parts before answering when accuracy matters.

**Escape untrusted text.** If a user's message contains `</document_content>`, it could break your structure, or try to inject instructions (m83). Escape `&`, `<` and `>` in anything you didn't write.

## 3. Templates, not string soup

```python
TEMPLATE = PromptTemplate("Summarize this for {audience}:\n<text>{text}</text>")
prompt = TEMPLATE.render(audience="a 12-year-old", text=article)
```

A template:

- lists its variables, and **fails loudly** on a missing or misspelled one (instead of sending `{audience}` literally to the model),
- lives in one place, so every call uses the same wording,
- has a **version** (a hash of its text), which you log with every call. When quality changes, you can tell which prompt version was running.

## 4. Getting parseable output

Ask for the answer inside a specific tag, then extract it:

```
Reply with the category inside <label></label> tags and nothing else.
```

Your parser should be forgiving where it's safe (whitespace, case) and strict where it matters (the label must be one of the allowed values; anything else becomes `"unknown"`, never a crash or a made-up category). For guaranteed JSON, use **structured outputs** (m78).

## 5. Evaluate prompt changes

A prompt change is a code change: it needs a test. Keep a small set of **test cases** (inputs with expected outputs), and measure accuracy for every prompt variant. Look at the **failures**, not just the score: they tell you what to fix next. Phase 8's m82 goes much deeper (LLM judges, confidence intervals); this module is the habit.

In the exercises, the fake model behaves like a real one in one important way: it only answers in the requested format when the prompt clearly asks for it. You'll see one prompt variant score 0% and another 100% on the same cases.

---

## Problem-solving habit #77: change one thing, measure, keep a log

When improving a prompt, change one thing at a time, run the test set, and write down the version, the change and the score. Prompt engineering without a log turns into superstition ("I think adding 'please' helped?").

## Common mistakes

- Testing a prompt on the one example you wrote it for.
- Building prompts with `+` and f-strings scattered across the codebase.
- Parsing replies with `reply.split(":")[1]` and crashing on the first unexpected answer.
- Pasting user input straight into the prompt without escaping or tags.
- Writing a wall of rules in capital letters instead of explaining the goal.

## Go deeper (optional)

1. Read Anthropic's prompt engineering guide (docs: "Prompt engineering overview", "Use XML tags", "Multishot prompting", "Long context tips"). Rewrite one of your prompts with what you learned and measure the difference.
2. Build a prompt that classifies the questions learners ask the mentor (concept / bug / exercise hint / other). Collect 30 real examples, label them, and iterate until accuracy stops improving. What were the biggest wins?
3. Read *Large Language Models are Zero-Shot Reasoners* (Kojima et al., 2022). Why does "Let's think step by step" matter less for models that think before answering?

## Your turn

Open the **Exercises** tab. Everything runs offline against the fake API.
