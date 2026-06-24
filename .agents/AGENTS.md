# Custom Agent Rules

You have a collection of custom skills installed. You MUST proactively use these skills whenever they are relevant to the user's prompt without the user needing to ask for them:

1. **Design & Frontend Tasks:** Whenever the user asks to build, design, refactor, style, or review a frontend interface, you MUST immediately invoke and read the appropriate design skill(s) before writing any code or proposing layout systems:
   - Use [design-taste-frontend](file:///C:/Users/Anon/.gemini/antigravity-ide/skills/design-taste-frontend/SKILL.md) for general landing pages/SaaS UIs.
   - Use [emil-design-eng](file:///C:/Users/Anon/.gemini/antigravity-ide/skills/emil-design-eng/SKILL.md) for UI polish, animations, and micro-interactions.
   - Use [high-end-visual-design](file:///C:/Users/Anon/.gemini/antigravity-ide/skills/high-end-visual-design/SKILL.md) for double-bezel card borders and agency-level aesthetics.
   - Use [industrial-brutalist-ui](file:///C:/Users/Anon/.gemini/antigravity-ide/skills/industrial-brutalist-ui/SKILL.md) for brutalist, monospaced, rigid grids, newsprint light-modes, or tactical dark-modes.
   - Use [minimalist-ui](file:///C:/Users/Anon/.gemini/antigravity-ide/skills/minimalist-ui/SKILL.md) for warm monochromatic, flat bento, muted pastel styles.
   - Use [stitch-design-taste](file:///C:/Users/Anon/.gemini/antigravity-ide/skills/stitch-design-taste/SKILL.md) when generating screen designs.
   - Use [redesign-existing-projects](file:///C:/Users/Anon/.gemini/antigravity-ide/skills/redesign-existing-projects/SKILL.md) for refactoring legacy UI code.

2. **Writing Code & Completeness:**
   - Always read and follow [full-output-enforcement](file:///C:/Users/Anon/.gemini/antigravity-ide/skills/full-output-enforcement/SKILL.md) for any code generation task to prevent partial code generation or placeholder patterns (`// ...`).

3. **Brevity & Token Optimization:**
   - If the user asks for brevity, mentions caveman, or invokes `/caveman`, immediately read and follow [caveman](file:///C:/Users/Anon/.gemini/antigravity-ide/skills/caveman/SKILL.md) (and utilize the [cavecrew](file:///C:/Users/Anon/.gemini/antigravity-ide/skills/cavecrew/SKILL.md) subagent presets for subagent delegation).

4. **Heavy Computations & Kaggle Offloading:**
   - Whenever execution requires running heavy scripts, sweeps, multi-model embedding evaluations, or large $N$ relational evaluations that could freeze or strain the user's local PC, you MUST design and package the experiments into self-contained, GPU-accelerated python scripts (formatted for single-cell Kaggle Notebook copy-paste) rather than proposing local terminal runs.

