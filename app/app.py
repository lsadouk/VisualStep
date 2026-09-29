"""
VisualStep — ADHD Classroom Instruction Decomposer
Gradio demo app for Hugging Face Spaces
"""

import json
import gradio as gr
import torch
import spaces

# ── Model loading ─────────────────────────────────────────────────────────────
MODEL_ID = "lsadouk1111/VisualStep"

model = None
tokenizer = None

# ── System prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are an expert in ADHD classroom accommodations for primary school children aged 5-10.
Transform the teacher's spoken classroom instruction into an ADHD-adapted visual step card.
Rules:
- Maximum 5 steps
- Each step: one concrete physical action only
- Each action: maximum 6 words
- Start each step with an imperative verb
- Include a pictogram keyword in the icon field
- Output JSON only, no other text
Output format:
{"steps":[{"id":1,"action":"Open your book","detail":"page 23","icon":"book","check":true}],"n_steps":1,"support_level":"medium"}
support_level: "light" (1-2 steps), "medium" (3-4 steps), "high" (5 steps)"""

# ── Icon to emoji ──────────────────────────────────────────────────────────────
EMOJI = {
    # core actions
    "book": "📖", "page": "📄", "pencil": "✏️", "write": "✏️", "writing": "✏️",
    "read": "👁️", "eye": "👁️", "hand": "✋", "finger": "👆",
    "talk": "🗣️", "listen": "👂", "think": "🧠", "look": "👁️", "show": "👁️",
    "draw": "🎨", "paper": "📝", "notebook": "📓", "board": "📋",
    "number": "🔢", "count": "🔢", "answer": "✅", "sort": "🗂️",
    "match": "🔗", "circle": "⭕", "plant": "🌱", "seed": "🌰",
    "water": "💧", "animal": "🐾", "picture": "🖼️", "image": "🖼️", "photo": "🖼️",
    "partner": "👥", "find": "🔍", "search": "🔍", "observe": "🔭",
    "measure": "📐", "ruler": "📏", "line": "📏",
    "record": "📊", "bar": "📊", "label": "🏷️",
    "colour": "🖌️", "color": "🖌️", "paintbrush": "🖌️",
    "cut": "✂️", "touch": "👆", "quiet": "🤫", "ready": "✅", "check": "✅",
    "open": "📂", "put": "📌", "place": "📌", "sit": "🪑", "chair": "🪑",
    "carpet": "🪑", "desk": "🪑",
    "science": "🔬", "maths": "➕", "english": "📚",
    # people / social
    "people": "👥", "two_people": "👥", "two-people": "👥", "friend": "👥",
    "group": "👥", "class": "🏫", "classroom": "🏫", "pairs": "👥", "share": "👥",
    # speech / thought
    "pause": "⏸️", "ear": "👂", "speak": "🗣️", "discussion": "🗣️",
    "buzz": "🗣️", "feedback": "💬", "speech_bubble": "💬", "speech-bubble": "💬",
    "thought": "💭", "thought-bubble": "💭", "thought_bubble": "💭",
    # writing tools
    "marker": "🖊️", "pen": "🖊️", "highlighter": "🖊️", "eraser": "✏️",
    "edit": "✏️", "underline": "📝", "sentence": "📝", "word": "📝",
    "text": "📝", "paragraph": "📝", "title": "📝", "story": "📖",
    # maths
    "math": "➕", "equation": "➕", "plus": "➕", "add": "➕", "addend": "➕",
    "divide": "➗", "calculator": "🔢", "blocks": "🧱", "build": "🧱",
    "model": "🧱", "pattern": "🔢", "seven": "7️⃣",
    # organising / tasks
    "select": "👆", "choose": "👆", "pick": "👆",
    "folder": "📁", "task": "📋", "tasks": "📋", "list": "📋", "work": "📋",
    "fact": "📋", "options": "📋", "details": "📋", "rules": "📋",
    "examples": "📋", "problem": "📋", "compare": "🔗", "checklist": "✅",
    "solve": "✅", "questions": "❓", "question": "❓",
    # ideas / feedback
    "idea": "💡", "lightbulb": "💡", "decision": "💡",
    # time / waiting
    "wait": "⏳", "clock": "⏰", "repeat": "🔁", "play": "▶️",
    # objects / materials
    "box": "📦", "materials": "📦", "coat": "🧥", "wire": "🔌",
    "cup": "🥛", "spoon": "🥄", "beaker": "🧪", "thermometer": "🌡️",
    "worksheet": "📝", "homework": "📝", "exercise": "📝",
    # nature / misc
    "tree": "🌳", "blueberry": "🫐", "nightshade": "🌿", "garden": "🌱",
    "cloud": "☁️", "fire": "🔥", "lightning": "⚡", "explosion": "💥",
    "light": "💡", "breathe": "🌬️", "matter": "🔬",
    # feedback / emotion
    "thumb": "👍", "happy": "😊", "heart": "❤️",
    # navigation
    "door": "🚪", "broken": "❌", "close": "❌",
    # misc
    "marble": "⚪", "period": "📝", "photo": "🖼️",
}

def get_emoji(icon):
    if not icon:
        return "📌"
    k = str(icon).lower()
    for key, em in EMOJI.items():
        if key in k:
            return em
    return "📌"

# ── Generation ────────────────────────────────────────────────────────────────
@spaces.GPU
def generate_step_card(instruction, year, subject):
    global model, tokenizer

    if not instruction.strip():
        return "Please enter a classroom instruction."

    # Load model on first call (inside GPU context)
    if model is None:
        from unsloth import FastModel
        
        model, tokenizer = FastModel.from_pretrained(
            model_name=MODEL_ID,
            max_seq_length=512,
            load_in_4bit=True,
            dtype=None,
        )
        FastModel.for_inference(model)
        model.eval()

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Year: {year} | Subject: {subject} | Instruction: {instruction}"},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = outputs[0][inputs["input_ids"].shape[1]:]
    raw = tokenizer.decode(generated, skip_special_tokens=True).strip()
    raw = raw.replace("```json", "").replace("```", "").strip()

    try:
        data = json.loads(raw)
        steps = data.get("steps", [])
        support = data.get("support_level", "medium")
        n = data.get("n_steps", len(steps))

        # Build visual card
        lines = ["## 📌 WHAT TO DO NOW\n"]
        for s in steps:
            emoji = get_emoji(s.get("icon", ""))
            action = s.get("action", "")
            detail = s.get("detail", "")
            line = f"☐ **{s.get('id', '')}.**  {emoji}  {action}"
            if detail:
                line += f" → *{detail}*"
            lines.append(line)

        lines.append(f"\n---")
        lines.append(f"*Support level: {support} · {n} step(s)*")

        return "\n\n".join(lines)

    except json.JSONDecodeError:
        return f"⚠️ Model output could not be parsed as JSON.\n\nRaw output:\n```\n{raw}\n```"

# ── Gradio interface ───────────────────────────────────────────────────────────
EXAMPLES = [
    ["Ok everyone, open your books to page 23, read the paragraph quietly, then answer questions 1 to 3 in your notebook and put your hand up when you're done.", "2", "english"],
    ["Right, so, um, can we all take our science books and turn to page 10? And then, after that, I want you to look at the picture and think about what you see.", "1", "science"],
    ["Everyone stop what you're doing, pack away your things, and line up quietly at the door please.", "3", "routine"],
    ["I want you to sort these animals into mammals and not mammals using the sorting hoops, then write one sentence explaining how you decided.", "1", "science"],
    ["OK so, find your maths book, open to page 15, do the first three problems and check your answers with your partner.", "2", "maths"],
]

with gr.Blocks(
    title="VisualStep — ADHD Instruction Decomposer",
    theme=gr.themes.Soft(primary_hue="blue"),
    css="""
        .card-output { font-size: 1.1em; line-height: 1.8; }
        h1 { color: #1F4E79; }
        .subtitle { color: #666; font-size: 0.95em; margin-top: -10px; }
    """
) as demo:

    gr.Markdown("""
# VisualStep
### ADHD-Adapted Visual Instruction Decomposition for Primary School Classrooms
*Fine-tuned Phi-3 Mini (3.8B) · Lamyaa Sadouk · EMSI Casablanca*
---
Enter a classroom instruction as a teacher would say it out loud.
VisualStep will decompose it into a structured, ADHD-adapted visual step card.
""")

    with gr.Row():
        with gr.Column(scale=1):
            instruction = gr.Textbox(
                label="Teacher's spoken instruction",
                placeholder="e.g. Ok everyone, open your books to page 23, read the paragraph quietly, then answer questions 1 to 3...",
                lines=4,
            )
            with gr.Row():
                year = gr.Dropdown(
                    choices=["1", "2", "3", "4", "5", "6"],
                    value="2",
                    label="Year group",
                )
                subject = gr.Dropdown(
                    choices=["english", "maths", "science", "routine", "transition"],
                    value="english",
                    label="Subject",
                )
            btn = gr.Button("Generate Step Card", variant="primary", size="lg")

        with gr.Column(scale=1):
            output = gr.Markdown(
                label="ADHD-adapted step card",
                elem_classes=["card-output"],
                value="*Your step card will appear here.*"
            )

    btn.click(
        fn=generate_step_card,
        inputs=[instruction, year, subject],
        outputs=output,
    )

    gr.Examples(
        examples=EXAMPLES,
        inputs=[instruction, year, subject],
        label="Try these examples",
    )

    gr.Markdown("""
---
**About VisualStep:**
This demo accompanies the paper *"VisualStep: A Fine-Tuned Small Language Model for ADHD-Adapted
Visual Instruction Decomposition in Primary School Classrooms"*.
The model was fine-tuned on VisualStep-2K, a dataset of 2,000 spoken classroom instruction–step card pairs.
All inference runs locally — no data is sent to external servers.
""")

demo.launch()
