# Post-Detection Processing Pipeline

## Input
Raw detection results from Grounding DINO + SAM2: a list of bounding boxes, each with an ID, type label, and coordinates.

## Output
A clean, hierarchical JSON tree representing the UI structure, with component patterns annotated.

---

## Step 1: Text Fragment Merging

**Goal**: Fix split text elements (e.g. "Game" + "Modes" → "Game Modes")

**Conditions for merging two boxes**:
- Both are text-type labels
- Horizontally adjacent: gap between them < 30% of the shorter box's width
- Vertically aligned: top coordinates differ by < 20% of box height
- Similar height: height difference < 20%

**Implementation**: Pure geometric rules. Iterate all text-type box pairs, merge qualifying pairs into a single box (union bounding rect), keep the type as text. Run iteratively until no more merges occur (handles 3+ fragment cases).

---

## Step 2: Layer Bleed-Through Removal

**Goal**: Remove elements that belong to background/obscured UI layers, not the current active page.

**Method — two-pass approach**:

**Pass 1 (geometric heuristics)**:
Flag boxes as suspicious if they meet ANY of:
- Bounding box edge is clipped by the screen boundary (one side flush with screen edge, opposite side looks unnaturally cut)
- Located outside the main content container (if a dominant large container box exists, elements outside it are suspicious)
- Very small area AND positioned at screen corners/edges

**Pass 2 (LLM confirmation — optional)**:
For flagged boxes, crop their region from the original image. Send to LLM with the question: "Is this UI element part of the active page or a background layer?" Remove confirmed bleed-through elements.

**Fallback**: If no clear main container exists, skip Pass 1 and rely on Pass 2 only. Or skip this step entirely and let the tree-building step naturally orphan these elements (they won't fit into any container), then prune orphans at the end.

---

## Step 3: Redundant Detection Removal (Overdraw Cleanup)

**Goal**: Remove duplicate detections of the same element.

**Condition for redundancy**: Two boxes have IoU > 0.85 AND their area ratio is between 0.8–1.2 (similar size). This distinguishes true duplicates from meaningful nesting (like button 28/42/37 where sizes differ significantly).

**Implementation**: Pure geometric rules. For each redundant pair, keep the one with the higher detection confidence score (from Grounding DINO). If confidence is similar, keep the larger box.

---

## Step 4: Containment Tree Construction

**Goal**: Build a parent-child hierarchy from the flat list of boxes.

**Containment rule**: Box B is a child of Box A if:
- B's area overlapping with A is ≥ 90% of B's total area
- A's area is at least 1.5× B's area (prevents near-same-size boxes from becoming parent-child)

**Building the tree**:
1. Sort all boxes by area, largest first
2. For each box (small to large), find its smallest qualifying parent — the smallest box that contains it per the rules above. This ensures each element is assigned to its most immediate parent, not a grandparent.
3. Boxes with no qualifying parent become root-level nodes

**Implementation**: Pure geometric rules. O(n²) pairwise comparison is fine for ~50 elements.

---

## Step 5: Ghost Parent Pruning

**Goal**: Remove nodes that have no semantic meaning and only exist as accidental containers.

**Condition**: A node is a ghost parent if:
- It has children
- Its type label is generic or absent (not a recognized container type like panel, label, card, button)
- Its bounding box is very close in size to the union of its children's bounding boxes (< 110% of children's combined extent) — meaning it adds no visible "background" or "padding" beyond its children

**Action**: Promote its children to its parent level, remove the ghost node.

**Implementation**: Post-order tree traversal (process leaves first, work upward). Pure rules.

---

## Step 6: Component Pattern Recognition

**Goal**: Identify repeated component structures among siblings, and infer layout direction.

**6a — Layout direction inference**:
For each set of siblings, determine arrangement:
- If their vertical centers are aligned (within 10% tolerance) → horizontal layout
- If their horizontal centers are aligned → vertical layout
- If both row and column patterns exist → grid layout
- Otherwise → absolute/free-form positioning

**Implementation**: Pure geometric rules based on center-point alignment.

**6b — Repeated component detection**:
Among siblings, identify groups that share the same structure:
- Same type label
- Similar aspect ratio (within 15%)
- Similar area (within 25%)
- Similar number of children
- Children have matching type label sequences

**Implementation**: Pairwise comparison of sibling nodes. Group matches together. Annotate each group with a `component_template` ID (e.g. "game_mode_card") and mark individual instances.

**6c — (Optional) LLM-assisted semantic labeling**:
For each identified component template, crop one representative instance from the original image. Send to LLM: "What kind of UI component is this? Give it a semantic name." This adds human-readable labels like "shop_offer_card" or "mode_selection_banner" instead of generic IDs.

---

## Step 7: Output Final Hierarchy

**Goal**: Export the clean tree as structured JSON for downstream use.

**Output format per node**:
```json
{
  "id": "node_18",
  "type": "label",
  "semantic_name": "game_mode_card",
  "bounds_pct": { "left": 2.1, "top": 22.5, "width": 95.8, "height": 18.3 },
  "layout": "horizontal",
  "component_template": "game_mode_card",
  "children": [ ... ]
}
```

Coordinates in percentage of parent bounds (for resolution independence).

**Implementation**: Tree serialization to JSON.

---

## Summary

| Step | What | Method | AI needed? |
|------|------|--------|------------|
| 1. Text merge | Fix split text boxes | Geometric rules | No |
| 2. Layer bleed-through | Remove background layer elements | Geometric rules + optional LLM | Optional |
| 3. Overdraw cleanup | Remove duplicate detections | Geometric rules (IoU + area ratio) | No |
| 4. Build tree | Establish parent-child hierarchy | Containment + area sorting | No |
| 5. Ghost pruning | Remove meaningless wrapper nodes | Post-order traversal + rules | No |
| 6. Pattern recognition | Find repeated components + layout | Geometric rules + optional LLM | Optional |
| 7. Export | Output clean hierarchy JSON | Serialization | No |

Most steps are pure geometric computation. LLM is only used optionally in steps 2 and 6 for semantic understanding, and can be skipped in early iterations.
