/**
 * week4/script.js — Association-rule mining starter (HW4).
 *
 * This is a plain (classic) script, NOT an ES module. `week4/transactions.js`
 * (loaded first, in a regular <script> tag) assigns the dictionary-encoded UCI
 * Online Retail baskets to `window.HW4`. This file reads that global, decodes
 * the integer-index baskets back into `{ stock, description }` items, renders a
 * dataset summary, and wires two slider controls ("minimum support" and
 * "minimum confidence") plus a "Run rules" button. Because nothing is fetched,
 * the page works from a `file://` URL with no server.
 *
 * Implemented student functions:
 *   1. `dedupeBasket`         — unique stock codes in a basket, first-appearance order.
 *   2. `countItemset`         — baskets containing every requested stock (`0` if any is absent).
 *   3. `computeSupport`       — support = count(A union B) / N, guarded when N = 0.
 *   4. `computeConfidence`    — confidence = count(A union B) / count(A), guarded when count(A) = 0.
 *   5. `computeLift`          — lift = confidence / (count(B) / N), guarded.
 *   6. `findFrequentItemsets` — mine all frequent itemsets with Apriori.
 *   7. `generateRules`        — turn frequent itemsets into both-direction rules.
 *
 * PROVIDED for you (scaffolding): dataset loading/decoding from `window.HW4`, the
 * inverted-index builder (`buildIndex` / `asIndex` / `indexCache`), the thin
 * counting wrappers `countItem` / `countPair` (they call `countItemset`),
 * threshold validation and slider
 * readout, the DOM wiring, all formatting and rendering helpers, and the test
 * harness. Leave the provided code as-is and implement only the stubs above.
 *
 * Metrics (see week4/readme.md for definitions):
 *   support(A -> B)    = count(A union B) / N
 *   confidence(A -> B) = count(A union B) / count(A)
 *   lift(A -> B)       = confidence(A -> B) / (count(B) / N)
 *
 * @module week4/script
 */

// Defensive guard: `transactions.js` must run before this file. If `window.HW4`
// is missing (for example because `transactions.js` was not copied next to
// `index.html`), fail loudly and visibly instead of throwing an opaque error.
if (typeof window === "undefined" || typeof window.HW4 === "undefined") {
  if (typeof document !== "undefined") {
    document.body.innerHTML =
      '<p style="padding:2rem;font-family:sans-serif">' +
      "Failed to load `transactions.js`. Make sure it sits next to `index.html` " +
      "and is loaded before `script.js`.</p>";
  }
  throw new Error(
    "window.HW4 is not defined — load transactions.js before script.js.",
  );
}

/** The raw dataset global assigned by `week4/transactions.js`. */
const data = window.HW4;

/**
 * Baskets decoded from the dictionary-encoded arrays in `window.HW4`. Each
 * basket is an array of `{ stock, description }` items, matching the fixture
 * shape used by the tests.
 *
 * @type {Array<Array<Item>>}
 */
const TRANSACTIONS = data.baskets.map((basket) =>
  basket.map((stockIndex) => ({
    stock: data.stocks[stockIndex],
    description: data.descriptions[stockIndex],
  })),
);

/**
 * Dataset-wide inverted index, built once at startup. `null` until `init()` has
 * run, so the UI can tell "still loading" from "loaded".
 *
 * @type {BasketIndex|null}
 */
let DATASET_INDEX = null;

/** Number of baskets (the `N` used by support and lift). */
let N = data.N_BASKETS;

/**
 * @typedef {Object} Item
 * @property {string} stock       Stock code (the item identity).
 * @property {string} description Human-readable product description.
 */

/**
 * @typedef {Object} Rule
 * @property {string[]} antecedent    Item identities on the left-hand side (A).
 * @property {string[]} consequent    Item identities on the right-hand side (B).
 * @property {number} jointCount      count(A union B).
 * @property {number} antecedentCount count(A).
 * @property {number} consequentCount count(B).
 * @property {number} support         jointCount / N.
 * @property {number} confidence      jointCount / antecedentCount.
 * @property {number} lift            confidence / (consequentCount / N).
 */

/**
 * @typedef {Object} BasketIndex
 * @property {number} n                              Number of baskets.
 * @property {Map<string, Set<number>>} byStock      Stock code -> basket ids.
 */

// ---------------------------------------------------------------------------
// Provided helpers and student stubs
//
// Functions carrying a `TODO(hw4)` marker are stubs you must implement; every
// other function in this file is scaffolding and should be left as-is.
// ---------------------------------------------------------------------------

/**
 * Read the item identity out of a basket entry. Accepts either a plain string or
 * an object with a `stock` property so the helpers work with both the real
 * dataset and the small fixtures used by the tests.
 *
 * @param {Item|string} item
 * @returns {string}
 */
function stockOf(item) {
  return typeof item === "string" ? item : item.stock;
}

/**
 * Build an inverted index from basket id to the set of baskets containing each
 * stock code. Counting a candidate itemset then becomes a set intersection.
 *
 * @param {Array<Array<Item|string>>} baskets
 * @returns {BasketIndex}
 */
function buildIndex(baskets) {
  /** @type {Map<string, Set<number>>} */
  const byStock = new Map();
  baskets.forEach((basket, basketId) => {
    for (const item of basket) {
      const stock = stockOf(item);
      let posting = byStock.get(stock);
      if (!posting) {
        posting = new Set();
        byStock.set(stock, posting);
      }
      posting.add(basketId);
    }
  });
  return { n: baskets.length, byStock };
}

/** Cache of lazily built indexes, keyed by the baskets array. */
const indexCache = new WeakMap();

/**
 * Accept either a ready-made index or a raw basket array and return an index.
 *
 * @param {BasketIndex|Array<Array<Item|string>>} basketsOrIndex
 * @returns {BasketIndex}
 */
function asIndex(basketsOrIndex) {
  if (basketsOrIndex && !Array.isArray(basketsOrIndex) && basketsOrIndex.byStock) {
    return basketsOrIndex;
  }
  let index = indexCache.get(basketsOrIndex);
  if (!index) {
    index = buildIndex(basketsOrIndex);
    indexCache.set(basketsOrIndex, index);
  }
  return index;
}

/**
 * Count the baskets that contain every stock code in `stocks`.
 *
 * Build (or reuse) a basket -> stock inverted index, intersect the
 * posting lists of the requested stocks, and return the size of the
 * intersection.
 *
 * Contract:
 *  - Accept either a ready-made index or a raw basket array. The provided
 *    `asIndex` helper returns an index for either input.
 *  - Return `0` when `stocks` is empty, and also when any requested stock is
 *    absent from the dataset (never throw for an unknown stock).
 *  - A stock repeated within one basket must be counted at most once. Dedupe the
 *    request before intersecting.
 *
 * @param {BasketIndex|Array<Array<Item|string>>} basketsOrIndex
 * @param {Array<string|Item>} stocks
 * @returns {number} count(A) for a single-element `stocks`, count(A union B) for two.
 */
function countItemset(basketsOrIndex, stocks) {
  const requested = dedupeBasket(stocks);
  if (requested.length === 0) return 0;

  // Posting sets are the non-zero rows of each binary invoice-item column:
  // duplicates within a basket still represent a single presence, not quantity.
  const index = asIndex(basketsOrIndex);
  const postings = [];
  for (const stock of requested) {
    const posting = index.byStock.get(stock);
    if (!posting || posting.size === 0) return 0;
    postings.push(posting);
  }
  postings.sort((a, b) => a.size - b.size);

  // Iterate the smallest column and test membership in every other column.
  // Only the local posting-list array is sorted; the index stays unchanged.
  let count = 0;
  for (const basketId of postings[0]) {
    if (postings.every((posting) => posting.has(basketId))) count++;
  }
  return count;
}

/**
 * Count the baskets containing a single stock code.
 *
 * @param {BasketIndex|Array<Array<Item|string>>} basketsOrIndex
 * @param {string|Item} stock
 * @returns {number}
 */
function countItem(basketsOrIndex, stock) {
  return countItemset(basketsOrIndex, [stock]);
}

/**
 * Count the baskets containing both `stockA` and `stockB`.
 *
 * @param {BasketIndex|Array<Array<Item|string>>} basketsOrIndex
 * @param {string|Item} stockA
 * @param {string|Item} stockB
 * @returns {number}
 */
function countPair(basketsOrIndex, stockA, stockB) {
  return countItemset(basketsOrIndex, [stockA, stockB]);
}

/**
 * Remove repeated item identities from a raw basket, keeping first-appearance
 * order. A basket is a set of items, so duplicates must not be counted twice.
 *
 * Walk the input once, map each entry to its stock code with the
 * provided `stockOf` helper, and return each distinct code the first time it
 * appears.
 *
 * Contract:
 *  - An empty input returns `[]`.
 *  - The input may mix plain strings and `{ stock, description }` objects.
 *  - Only the stock code is returned; the description is dropped.
 *
 * @param {Array<string|Item>} rawItems
 * @returns {Array<string>} unique stock codes, in first-appearance order.
 */
function dedupeBasket(rawItems) {
  const seen = new Set();
  const stocks = [];
  for (const item of rawItems) {
    const stock = stockOf(item);
    if (!seen.has(stock)) {
      seen.add(stock);
      stocks.push(stock);
    }
  }
  return stocks;
}

/**
 * Compute support as `jointCount / n`.
 *
 * Return the fraction and flag the `n === 0` case.
 *
 * Contract:
 *  - `defined: true` with `value = jointCount / n` when `n > 0`.
 *  - `defined: false` with `value = 0` when `n === 0` (never divide by zero).
 *
 * @param {number} jointCount count(A union B)
 * @param {number} n          number of baskets
 * @returns {{value: number, defined: boolean}} `defined` is false when `n === 0`.
 */
function computeSupport(jointCount, n) {
  if (n === 0) return { value: 0, defined: false };
  return { value: jointCount / n, defined: true };
}

/**
 * Compute confidence as `jointCount / antecedentCount`.
 *
 * Return the fraction and flag the `antecedentCount === 0` case.
 *
 * Contract:
 *  - `defined: true` with `value = jointCount / antecedentCount` when count(A) > 0.
 *  - `defined: false` with `value = 0` when count(A) === 0: a rule whose
 *    left-hand side never occurs has no confidence.
 *
 * @param {number} jointCount      count(A union B)
 * @param {number} antecedentCount count(A)
 * @returns {{value: number, defined: boolean}}
 */
function computeConfidence(jointCount, antecedentCount) {
  if (antecedentCount === 0) return { value: 0, defined: false };
  return { value: jointCount / antecedentCount, defined: true };
}

/**
 * Compute lift as `confidence / (consequentCount / n)`.
 *
 * Divide the incoming confidence by the consequent's baseline rate.
 *
 * Contract:
 *  - `defined: true` with `value = confidence.value / (consequentCount / n)`
 *    when every input is usable.
 *  - `defined: false` with `value = 0` when the incoming confidence is missing or
 *    undefined, when `n === 0`, or when the baseline `consequentCount / n` is 0
 *    (the consequent never occurs).
 *
 * @param {{value: number, defined: boolean}} confidence confidence(A -> B)
 * @param {number} consequentCount count(B)
 * @param {number} n               number of baskets
 * @returns {{value: number, defined: boolean}}
 */
function computeLift(confidence, consequentCount, n) {
  if (!confidence || !confidence.defined || n === 0 || consequentCount === 0) {
    return { value: 0, defined: false };
  }
  return { value: confidence.value / (consequentCount / n), defined: true };
}

/**
 * Validate the two slider values, expressed as fractions in `(0, 1]`.
 *
 * @param {number} minSupport    minimum support fraction
 * @param {number} minConfidence minimum confidence fraction
 * @returns {{ok: boolean, errors: string[]}}
 */
function validateThresholds(minSupport, minConfidence) {
  const errors = [];
  for (const [label, value] of [
    ["Minimum support", minSupport],
    ["Minimum confidence", minConfidence],
  ]) {
    if (typeof value !== "number" || Number.isNaN(value)) {
      errors.push(`${label} must be a number.`);
    } else if (value <= 0 || value > 1) {
      errors.push(`${label} must be greater than 0 and at most 1.`);
    }
  }
  return { ok: errors.length === 0, errors };
}

/**
 * Mine all frequent itemsets whose support is at least `minSupport`.
 *
 * Apriori uses level-wise prefix joins and downward-closure pruning. Occurrence
 * counts come from posting-list intersections, never a full scan per candidate.
 * It is chosen for its inspectable join/prune/count steps; FP-Growth avoids
 * candidate enumeration but needs an additional prefix-tree representation.
 *
 * Contract:
 *  - Return one entry per frequent itemset: `{ items, count, support }`, where
 *    `items` is the (deduplicated) set of stock codes, `count` is the number of
 *    baskets containing every item, and `support = count / N`.
 *  - Every returned itemset must satisfy `support >= minSupport`.
 *  - You may call `countItemset` while developing, but a per-candidate scan over
 *    17,080 baskets is slow — build your own occurrence/index structures.
 *  - `generateRules` consumes this exact shape, so keep the field names stable.
 *
 * @param {Array<Array<Item|string>>} transactions baskets
 * @param {number} minSupport minimum support fraction in `(0, 1]`
 * @returns {Array<{items: string[], count: number, support: number}>} frequent itemsets
 */
function findFrequentItemsets(transactions, minSupport) {
  const index = asIndex(transactions);
  if (index.n === 0) return [];
  const keyOf = (items) => JSON.stringify(items);

  // Canonical StockCode order is code-unit order, independent of locale.
  let level = [...index.byStock.keys()].sort().flatMap((stock) => {
    const count = index.byStock.get(stock).size;
    const support = count / index.n;
    return support >= minSupport ? [{ items: [stock], count, support }] : [];
  });
  const frequent = [...level];

  while (level.length > 0) {
    const width = level[0].items.length;
    const previousKeys = new Set(level.map((entry) => keyOf(entry.items)));
    const groups = new Map();
    for (const entry of level) {
      const prefix = keyOf(entry.items.slice(0, width - 1));
      if (!groups.has(prefix)) groups.set(prefix, []);
      groups.get(prefix).push(entry);
    }

    const next = [];
    for (const group of groups.values()) {
      for (let i = 0; i < group.length; i++) {
        for (let j = i + 1; j < group.length; j++) {
          const candidate = [...group[i].items, group[j].items[width - 1]];
          // Every width-item subset must be frequent before counting this join.
          const allSubsetsFrequent = candidate.every((_, omitted) =>
            previousKeys.has(keyOf(candidate.filter((_, p) => p !== omitted))),
          );
          if (!allSubsetsFrequent) continue;

          const count = countItemset(index, candidate);
          const support = count / index.n;
          if (support >= minSupport) next.push({ items: candidate, count, support });
        }
      }
    }
    // Prefix groups and their increasing suffixes produce unique sorted joins.
    for (const entry of next) frequent.push(entry);
    level = next;
  }
  return frequent;
}

/**
 * Turn frequent itemsets into association rules and keep the ones whose
 * confidence is at least `minConfidence`.
 *
 * For each frequent itemset, split it into a non-empty antecedent `A`
 * and a non-empty, disjoint consequent `B` in BOTH directions, compute the
 * confidence for each direction, and keep the rules that pass the threshold.
 *
 * Contract:
 *  - Each returned rule follows the `Rule` shape documented at the top of this
 *    module. At minimum it carries `antecedent` and `consequent`; the renderer
 *    fills in the counts and metrics with `enrichRule`, but returning them
 *    yourself is fine and faster.
 *  - Generate both `A -> B` and `B -> A`: they are separate rules with (usually)
 *    different confidence. Skip a direction whose consequent is empty.
 *  - Keep only rules with `confidence >= minConfidence`. `count(A)` is non-zero
 *    for every generated rule, so the confidence is always defined.
 *
 * @param {Array<{items: string[], count: number, support: number}>} frequentItemsets
 * @param {number} minConfidence minimum confidence fraction in `(0, 1]`
 * @returns {Rule[]}
 */
function generateRules(frequentItemsets, minConfidence) {
  const keyOf = (items) => JSON.stringify(items);
  const byItems = new Map();
  for (const entry of frequentItemsets) {
    const items = dedupeBasket(entry.items).sort();
    if (items.length > 0) byItems.set(keyOf(items), { ...entry, items });
  }
  const compareItems = (a, b) => {
    if (a.items.length !== b.items.length) return a.items.length - b.items.length;
    for (let i = 0; i < a.items.length; i++) {
      if (a.items[i] !== b.items[i]) return a.items[i] < b.items[i] ? -1 : 1;
    }
    return 0;
  };
  const itemsets = [...byItems.values()].sort(compareItems);
  const rules = [];

  for (const joint of itemsets) {
    if (joint.items.length < 2) continue;
    const antecedent = [];
    const consequent = [];
    // Recursive partitions avoid JavaScript's 32-bit bitmask length limit.
    // Each item is assigned to A or B: exactly 2^k - 2 non-empty splits.
    const visit = (position) => {
      if (position < joint.items.length) {
        antecedent.push(joint.items[position]);
        visit(position + 1);
        antecedent.pop();
        consequent.push(joint.items[position]);
        visit(position + 1);
        consequent.pop();
        return;
      }
      if (antecedent.length === 0 || consequent.length === 0) return;
      const left = byItems.get(keyOf(antecedent));
      const right = byItems.get(keyOf(consequent));
      if (!left || !right) {
        throw new Error("Frequent itemsets must include every non-empty subset.");
      }
      const confidence = computeConfidence(joint.count, left.count);
      if (!confidence.defined || confidence.value < minConfidence) return;
      // right.support is count(B)/N of the SAME input sample. No global N,
      // dataset scans, or N reconstructed through floating-point division.
      rules.push({
        antecedent: [...antecedent],
        consequent: [...consequent],
        jointCount: joint.count,
        antecedentCount: left.count,
        consequentCount: right.count,
        support: joint.support,
        confidence: confidence.value,
        lift: confidence.value / right.support,
      });
    };
    visit(0);
  }
  // Return ALL support/confidence-qualified rules. The analysis layer stores
  // rules.filter(rule => rule.lift > 1) separately as business candidates.
  return rules;
}

// ---------------------------------------------------------------------------
// Tiny worked example (used by the automated tests and the readout panel)
// ---------------------------------------------------------------------------

/**
 * A five-basket fixture with hand-computed support / confidence / lift values.
 *
 * The fixture is deliberately small enough to check with a pencil:
 *
 *   T1: bread, milk, jam, ham
 *   T2: bread, milk, jam
 *   T3: bread, milk
 *   T4: bread, jam, eggs
 *   T5: bread, eggs
 *
 * Item counts: bread = 5, milk = 3, jam = 3, eggs = 2, ham = 1 (N = 5).
 *
 * Because `bread` appears in every basket, every rule with `bread` on either
 * side has lift exactly 1 — that is the teaching point of the fixture.
 *
 * @returns {{n: number, baskets: string[][], itemCounts: Object<string, number>, rules: Array<Object>}}
 */
function tinyWorkedExample() {
  const baskets = [
    ["bread", "milk", "jam", "ham"],
    ["bread", "milk", "jam"],
    ["bread", "milk"],
    ["bread", "jam", "eggs"],
    ["bread", "eggs"],
  ];
  const n = baskets.length;
  return {
    n,
    baskets,
    itemCounts: { bread: 5, milk: 3, jam: 3, eggs: 2, ham: 1 },
    rules: [
      // lift == 1: bread is in every basket, so confidence equals support(B).
      {
        antecedent: ["bread"],
        consequent: ["milk"],
        jointCount: 3,
        antecedentCount: 5,
        consequentCount: 3,
        support: 3 / n,
        confidence: 3 / 5,
        lift: 1,
      },
      // lift > 1: milk and jam co-occur more than independence predicts.
      {
        antecedent: ["milk"],
        consequent: ["jam"],
        jointCount: 2,
        antecedentCount: 3,
        consequentCount: 3,
        support: 2 / n,
        confidence: 2 / 3,
        lift: (2 / 3) / (3 / 5),
      },
      // lift < 1 (and not degenerate): jam and eggs co-occur less than expected.
      {
        antecedent: ["jam"],
        consequent: ["eggs"],
        jointCount: 1,
        antecedentCount: 3,
        consequentCount: 2,
        support: 1 / n,
        confidence: 1 / 3,
        lift: (1 / 3) / (2 / 5),
      },
    ],
  };
}

// ---------------------------------------------------------------------------
// Rule normalisation
// ---------------------------------------------------------------------------

/**
 * Fill in any missing counts / metrics on a rule using the dataset index, so the
 * table can render a rule even when the student returns only `antecedent` and
 * `consequent`.
 *
 * @param {Partial<Rule>} rule
 * @param {BasketIndex} index
 * @returns {Rule}
 */
function enrichRule(rule, index) {
  const antecedent = (rule.antecedent || []).map(stockOf);
  const consequent = (rule.consequent || []).map(stockOf);
  const n = index.n;
  const jointCount =
    typeof rule.jointCount === "number"
      ? rule.jointCount
      : countItemset(index, [...antecedent, ...consequent]);
  const antecedentCount =
    typeof rule.antecedentCount === "number"
      ? rule.antecedentCount
      : countItemset(index, antecedent);
  const consequentCount =
    typeof rule.consequentCount === "number"
      ? rule.consequentCount
      : countItemset(index, consequent);
  const support = computeSupport(jointCount, n);
  const confidence = computeConfidence(jointCount, antecedentCount);
  const lift = computeLift(confidence, consequentCount, n);
  return {
    antecedent,
    consequent,
    jointCount,
    antecedentCount,
    consequentCount,
    support: typeof rule.support === "number" ? rule.support : support.value,
    confidence:
      typeof rule.confidence === "number" ? rule.confidence : confidence.value,
    lift: typeof rule.lift === "number" ? rule.lift : lift.value,
    supportDefined: support.defined,
    confidenceDefined: confidence.defined,
    liftDefined: lift.defined,
  };
}

/**
 * Swap the antecedent and consequent of a rule and recompute the metrics.
 *
 * Confidence is not symmetric, so `B -> A` usually has a different confidence.
 * Support and lift are unchanged because the joint itemset is the same.
 *
 * @param {Rule} rule
 * @param {BasketIndex} index
 * @returns {Rule}
 */
function reverseRule(rule, index) {
  return enrichRule(
    {
      antecedent: rule.consequent,
      consequent: rule.antecedent,
    },
    index,
  );
}

// ---------------------------------------------------------------------------
// Formatting helpers
// ---------------------------------------------------------------------------

/**
 * Format a fraction as a percentage with two decimals.
 *
 * @param {number} fraction
 * @returns {string}
 */
function formatPercent(fraction) {
  return `${(fraction * 100).toFixed(2)}%`;
}

/**
 * Format a number with four significant decimals for the results table.
 *
 * @param {number} value
 * @returns {string}
 */
function formatMetric(value) {
  if (!Number.isFinite(value)) return "n/a";
  return value.toFixed(4);
}

/**
 * Render a list of stock codes as readable text.
 *
 * @param {string[]} stocks
 * @param {BasketIndex} index
 * @returns {string}
 */
function formatItemset(stocks, index) {
  if (!stocks || stocks.length === 0) return "(empty)";
  return stocks.map((stock) => describe(stock, index)).join(", ");
}

/**
 * Lookup a stock code's canonical description, falling back to the code. Filled
 * by `primeDescriptions()` once `init()` runs.
 */
const descriptionByStock = new Map();

/**
 * Render `STOCK — human readable description` for one item.
 *
 * @param {string} stock
 * @param {BasketIndex} index
 * @returns {string}
 */
function describe(stock, index) {
  const description = descriptionByStock.get(stock);
  return description ? `${stock} — ${description}` : stock;
}

// ---------------------------------------------------------------------------
// Rendering
// ---------------------------------------------------------------------------

// Presentation state only. The miner's complete output is never sorted in place,
// filtered for display in the algorithm, or truncated to the page size.
const RULE_VIEW = {
  rules: [], entries: [], index: null, hasRun: false, page: 0, pageSize: 25,
  query: "", sort: "confidence", descending: true, selected: null,
  thresholds: null, running: false,
};

function itemLines(stocks, index) {
  return stocks.map((stock) =>
    `<span class="item-line"><span class="stock-tag">${escapeHtml(stock)}</span> — ${escapeHtml(descriptionByStock.get(stock) || stock)}</span>`,
  ).join("");
}

function setStatus(state, message) {
  const target = document.getElementById("status");
  if (target) { target.dataset.state = state; target.textContent = message; }
}

function clearRuleSelection() {
  RULE_VIEW.selected = null;
  const detail = document.getElementById("rule-detail");
  if (detail) detail.innerHTML = '<p class="empty-state">Choose a rule to see its counts, baseline and reverse direction.</p>';
}

function updateSelectedRows() {
  document.querySelectorAll("#results tr[data-rule-index]").forEach((row) => {
    const selected = Number(row.dataset.ruleIndex) === RULE_VIEW.selected;
    row.setAttribute("aria-selected", String(selected));
    const label = row.querySelector(".selection-label");
    if (label) label.hidden = !selected;
  });
}

/**
 * Render the dataset summary at the top of the page: total baskets, distinct
 * items, and the top five items by basket count.
 *
 * @param {BasketIndex} index
 * @param {HTMLElement|null} container
 * @returns {void}
 */
function renderDatasetSummary(index, container) {
  const target = container || document.getElementById("dataset-summary-body");
  if (!target) return;

  const counts = [...index.byStock.entries()]
    .map(([stock, posting]) => ({ stock, count: posting.size }))
    .sort((a, b) => b.count - a.count || a.stock.localeCompare(b.stock));
  const topFive = counts.slice(0, 5);

  const rows = topFive
    .map(
      (entry, i) =>
        `<tr><td class="num">${i + 1}</td><td>${escapeHtml(
          describe(entry.stock, index),
        )}</td><td class="num">${entry.count}</td></tr>`,
    )
    .join("");

  target.innerHTML = `<div class="dataset-strip">
    <div class="dataset-stat"><strong>${index.n.toLocaleString("en-US")}</strong><span>invoice baskets</span></div>
    <div class="dataset-stat"><strong>${index.byStock.size.toLocaleString("en-US")}</strong><span>distinct StockCodes</span></div>
    <span class="dataset-caption">UCI Online Retail · presence data · no timestamps</span>
  </div>`;
  const provenance = document.getElementById("dataset-provenance");
  if (provenance) provenance.textContent = data.dataset_provenance;
  const topItems = document.getElementById("top-items");
  if (topItems) topItems.innerHTML = `<h3>Top 5 items by basket count</h3>
    <div class="table-scroll"><table class="data-table"><thead><tr><th scope="col">#</th><th scope="col">Item</th><th scope="col">Baskets</th></tr></thead><tbody>${rows}</tbody></table></div>`;
}

/**
 * Render the rule list into a table. Each row is clickable and shows the rule in
 * the detail panel.
 *
 * @param {Rule[]} rules
 * @param {BasketIndex} [index]
 * @param {HTMLElement|null} [container]
 * @returns {void}
 */
function renderResults(rules, index, container) {
  const target = container || document.getElementById("results");
  if (!target) return;
  // The unchanged self-check harness renders an empty array in a scratch div.
  if (target !== document.getElementById("results")) {
    target.innerHTML = !rules || !rules.length ?
      '<p class="empty-state">No rules passed the current thresholds. ' +
      "Lower the minimum support or confidence and run again.</p>" :
      `<table class="data-table"><tbody>${rules.map((rule) => {
        const entry = enrichRule(rule, index || DATASET_INDEX);
        return `<tr><td>${escapeHtml(formatItemset(entry.antecedent,index))} → ${escapeHtml(formatItemset(entry.consequent,index))}</td><td>${entry.jointCount}</td></tr>`;
      }).join("")}</tbody></table>`;
    return;
  }
  RULE_VIEW.rules = rules || [];
  RULE_VIEW.index = index || DATASET_INDEX;
  RULE_VIEW.hasRun = true;
  RULE_VIEW.page = 0;
  RULE_VIEW.entries = RULE_VIEW.rules.map((rule, position) => ({
    position, rule: enrichRule(rule, RULE_VIEW.index),
    search: formatItemset([...rule.antecedent, ...rule.consequent], RULE_VIEW.index).toLowerCase(),
  }));
  clearRuleSelection();
  renderRulePage();
}

function visibleRuleEntries() {
  const terms = RULE_VIEW.query.trim().toLowerCase().split(/\s+/).filter(Boolean);
  const entries = RULE_VIEW.entries.filter((entry) => terms.every((term) => entry.search.includes(term)));
  const field = RULE_VIEW.sort;
  const direction = RULE_VIEW.descending ? -1 : 1;
  return entries.sort((a, b) => {
    const flag = field === "jointCount" ? null : `${field}Defined`;
    const aDefined = (!flag || a.rule[flag]) && Number.isFinite(a.rule[field]);
    const bDefined = (!flag || b.rule[flag]) && Number.isFinite(b.rule[field]);
    if (aDefined !== bDefined) return aDefined ? -1 : 1;
    return (aDefined ? direction * (a.rule[field] - b.rule[field]) : 0) || a.position - b.position;
  });
}

function renderRulePage() {
  const target = document.getElementById("results");
  if (!target) return;
  const entries = visibleRuleEntries();
  const pages = Math.max(1, Math.ceil(entries.length / RULE_VIEW.pageSize));
  RULE_VIEW.page = Math.max(0, Math.min(RULE_VIEW.page, pages - 1));
  const start = RULE_VIEW.page * RULE_VIEW.pageSize;
  const shown = entries.slice(start, start + RULE_VIEW.pageSize);
  const summary = document.getElementById("results-summary");
  if (summary) summary.textContent = `${RULE_VIEW.rules.length.toLocaleString("en-US")} found · ${entries.length.toLocaleString("en-US")} search ${entries.length === 1 ? "match" : "matches"}` +
    (entries.length ? ` · showing ${start + 1}–${start + shown.length}` : "");
  if (!entries.length) {
    target.innerHTML = RULE_VIEW.rules.length
      ? '<p class="empty-state">No matches for this search. Try a StockCode, a product name, or clear the search.</p>'
      : '<p class="empty-state">No rules passed the current thresholds. Lower support or confidence and run again.</p>';
  } else {
    const metric = (rule, name, percent) => rule[`${name}Defined`]
      ? (percent ? formatPercent(rule[name]) : formatMetric(rule[name])) : "undefined";
    target.innerHTML = `<div class="table-scroll" tabindex="0" aria-label="Rules table; scroll to view all columns">
      <table class="data-table rules-table"><caption class="sr-only">${RULE_VIEW.rules.length} mined rules; ${entries.length} search matches. Page ${RULE_VIEW.page + 1} of ${pages}.</caption>
      <thead><tr><th scope="col">Rule · A → B</th><th scope="col">Joint baskets</th><th scope="col">Support %</th><th scope="col">Confidence %</th><th scope="col">Lift ×</th></tr></thead>
      <tbody>${shown.map(({rule, position}) => `<tr tabindex="0" data-rule-index="${position}" aria-selected="${RULE_VIEW.selected === position}" aria-label="Inspect rule ${escapeHtml(rule.antecedent.join(', '))} to ${escapeHtml(rule.consequent.join(', '))}">
        <td class="rule-label"><span class="selection-label" ${RULE_VIEW.selected === position ? "" : "hidden"}>Selected</span><span class="side-label">A</span>${itemLines(rule.antecedent, RULE_VIEW.index)}
          <span class="rule-arrow" aria-hidden="true">→</span><span class="side-label">B</span>${itemLines(rule.consequent, RULE_VIEW.index)}</td>
        <td class="num">${rule.jointCount.toLocaleString("en-US")}</td><td class="num">${metric(rule,"support",true)}</td>
        <td class="num">${metric(rule,"confidence",true)}</td><td class="num">${metric(rule,"lift",false)}</td></tr>`).join("")}</tbody></table></div>`;
    target.querySelectorAll("tr[data-rule-index]").forEach((row) => {
      const activate = () => {
        RULE_VIEW.selected = Number(row.dataset.ruleIndex);
        updateSelectedRows();
        renderRuleDetail(RULE_VIEW.rules[RULE_VIEW.selected], RULE_VIEW.index);
        if (window.matchMedia("(max-width:980px)").matches) {
          document.getElementById("rule-detail").scrollIntoView({block:"start"});
          document.getElementById("detail-heading").focus({preventScroll:true});
        }
      };
      row.addEventListener("click", activate);
      row.addEventListener("keydown", (event) => {
        if (event.key === "Enter" || event.key === " ") { event.preventDefault(); activate(); }
      });
    });
  }
  const pagination = document.getElementById("pagination");
  if (pagination) pagination.hidden = !entries.length;
  const label = document.getElementById("page-label");
  if (label) label.textContent = `Page ${RULE_VIEW.page + 1} of ${pages} · ${RULE_VIEW.pageSize} per page`;
  const prev = document.getElementById("page-prev");
  const next = document.getElementById("page-next");
  if (prev) prev.disabled = RULE_VIEW.page === 0;
  if (next) next.disabled = RULE_VIEW.page >= pages - 1;
}

/**
 * Render the detail panel for one selected rule, including a "Reverse direction"
 * button that swaps A and B and re-renders the panel.
 *
 * Shows an inline note when `count(A)` or `count(B)` is zero, because confidence
 * and lift are then undefined.
 *
 * @param {Rule} rule
 * @param {BasketIndex} [index]
 * @param {HTMLElement|null} [container]
 * @returns {void}
 */
function renderRuleDetail(rule, index, container) {
  const target = container || document.getElementById("rule-detail");
  if (!target) return;
  const activeIndex = index || DATASET_INDEX;
  const enriched = enrichRule(rule, activeIndex);
  const reversed = reverseRule(enriched, activeIndex);

  const zeroDenominatorNotes = [];
  if (activeIndex.n === 0) {
    zeroDenominatorNotes.push(
      "N = 0, so support and lift are undefined (empty dataset).",
    );
  }
  if (enriched.antecedentCount === 0) {
    zeroDenominatorNotes.push(
      "count(A) = 0, so confidence(A → B) is undefined (division by zero); lift is undefined too.",
    );
  }
  if (enriched.consequentCount === 0) {
    zeroDenominatorNotes.push(
      "count(B) = 0, so lift(A → B) is undefined (division by zero).",
    );
  }
  const noteHtml = zeroDenominatorNotes.length
    ? `<p class="warning">${zeroDenominatorNotes.map(escapeHtml).join(" ")}</p>`
    : "";

  const baseline = computeSupport(enriched.consequentCount, activeIndex.n);
  const reverseBaseline = computeSupport(reversed.consequentCount, activeIndex.n);
  const show = (r, field, percent = true) => r[`${field}Defined`]
    ? (percent ? formatPercent(r[field]) : formatMetric(r[field])) : "undefined";
  const interpretation = enriched.confidenceDefined && baseline.defined && enriched.liftDefined
    ? `In ${enriched.jointCount.toLocaleString("en-US")} of ${enriched.antecedentCount.toLocaleString("en-US")} baskets with A, B also appears (${formatPercent(enriched.confidence)}). B appears in ${formatPercent(baseline.value)} of all baskets. The observed co-occurrence ratio is ${formatMetric(enriched.lift)}× the independence baseline.`
    : "One or more metrics are undefined because the dataset or a rule side has no observations. Undefined is not a measured zero association.";
  target.innerHTML = `<div class="detail-direction"><span class="side-label">Antecedent · A</span>${itemLines(enriched.antecedent,activeIndex)}
      <p class="detail-arrow" aria-label="association direction">→</p><span class="side-label">Consequent · B</span>${itemLines(enriched.consequent,activeIndex)}</div>
    <div class="metric-tiles">
      <div class="metric-tile"><span>Support</span><strong data-metric="support">${show(enriched,"support")}</strong><small>Joint baskets / all baskets</small></div>
      <div class="metric-tile"><span>Confidence</span><strong data-metric="confidence">${show(enriched,"confidence")}</strong><small>Joint baskets / baskets with A</small></div>
      <div class="metric-tile"><span>Lift · coefficient</span><strong data-metric="lift">${show(enriched,"lift",false)}${enriched.liftDefined ? "×" : ""}</strong><small>Confidence / baseline P(B)</small></div>
      <div class="metric-tile"><span>Baseline · P(B)</span><strong data-metric="baseline">${baseline.defined ? formatPercent(baseline.value) : "undefined"}</strong><small>Baskets with B / all baskets</small></div>
    </div>
    <dl class="count-grid"><div><dt>N · baskets</dt><dd data-count="N">${activeIndex.n.toLocaleString("en-US")}</dd></div>
      <div><dt>count(A)</dt><dd data-count="A">${enriched.antecedentCount.toLocaleString("en-US")}</dd></div>
      <div><dt>count(B)</dt><dd data-count="B">${enriched.consequentCount.toLocaleString("en-US")}</dd></div>
      <div><dt>count(A∪B)</dt><dd data-count="joint">${enriched.jointCount.toLocaleString("en-US")}</dd></div></dl>
    <p class="interpretation">${escapeHtml(interpretation)}</p>
    ${noteHtml}
    <h3 class="reverse-heading">Compare directions</h3>
    <table class="data-table comparison-table"><caption class="sr-only">Both directions, using the same dataset</caption>
      <thead><tr><th scope="col">Metric</th><th scope="col">A → B</th><th scope="col">B → A</th></tr></thead><tbody>
      <tr><th scope="row">Support</th><td>${show(enriched,"support")}</td><td>${show(reversed,"support")}</td></tr>
      <tr><th scope="row">Confidence</th><td>${show(enriched,"confidence")}</td><td>${show(reversed,"confidence")}</td></tr>
      <tr><th scope="row">Lift ×</th><td>${show(enriched,"lift",false)}</td><td>${show(reversed,"lift",false)}</td></tr>
      <tr><th scope="row">Right-side baseline</th><td>${baseline.defined ? formatPercent(baseline.value) : "undefined"}</td><td>${reverseBaseline.defined ? formatPercent(reverseBaseline.value) : "undefined"}</td></tr>
      </tbody></table>
    <p class="caveat">Confidence may change with direction; support and lift do not. The reverse may fail your confidence threshold. Lift is not a probability, a significance test or a proven sales increase.</p>
    <button type="button" id="reverse-rule" class="secondary">Reverse direction · B → A</button>`;

  const button = target.querySelector("#reverse-rule");
  button.addEventListener("click", () => {
    renderRuleDetail(reversed, activeIndex, target);
    target.querySelector("#reverse-rule").focus({preventScroll:true});
  });
}

/**
 * Render the optional worked-example readout panel.
 *
 * @param {ReturnType<typeof tinyWorkedExample>} example
 * @param {HTMLElement|null} [container]
 * @returns {void}
 */
function renderWorkedExample(example, container) {
  const target = container || document.getElementById("worked-example");
  if (!target) return;
  const rows = example.rules
    .map(
      (rule) => `
      <tr>
        <td>${rule.antecedent.join(", ")}</td>
        <td>${rule.consequent.join(", ")}</td>
        <td class="num">${rule.jointCount}</td>
        <td class="num">${rule.antecedentCount}</td>
        <td class="num">${rule.consequentCount}</td>
        <td class="num">${formatPercent(rule.support)}</td>
        <td class="num">${formatPercent(rule.confidence)}</td>
        <td class="num">${formatMetric(rule.lift)}</td>
      </tr>`,
    )
    .join("");
  target.innerHTML = `
    <p class="worked-example-intro">
      Five hand-built baskets, N = ${example.n}. The top row has lift exactly 1
      because <code>bread</code> appears in every basket; the second has lift &gt; 1;
      the third has lift &lt; 1.
    </p>
    <table class="data-table">
      <thead>
        <tr>
          <th scope="col">A</th><th scope="col">B</th>
          <th scope="col">count(A∪B)</th><th scope="col">count(A)</th><th scope="col">count(B)</th>
          <th scope="col">support</th><th scope="col">confidence</th><th scope="col">lift</th>
        </tr>
      </thead>
      <tbody>${rows}</tbody>
    </table>`;
}

/**
 * Escape text before inserting it into HTML.
 *
 * @param {string} text
 * @returns {string}
 */
function escapeHtml(text) {
  return String(text)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

// ---------------------------------------------------------------------------
// Test harness
// ---------------------------------------------------------------------------

/** Marker thrown by unimplemented `TODO(hw4)` stubs. */
const TODO_MARKER = "TODO(hw4)";

/**
 * Run the automated self-checks against the five-basket worked example and the
 * reference helpers. Unimplemented `TODO(hw4)` functions are reported as
 * `pending` rather than `fail`, so the harness is useful before and after the
 * assignment is implemented.
 *
 * @param {HTMLElement|null} [logElement] element that receives the text log
 * @returns {{passed: number, failed: number, pending: number, checks: Array<Object>}}
 */
function runTests(logElement) {
  const example = tinyWorkedExample();
  const checks = [];
  const basketObjects = example.baskets;

  const pass = (name, detail) => checks.push({ name, status: "PASS", detail });
  const fail = (name, detail) => checks.push({ name, status: "FAIL", detail });
  const pending = (name, detail) => checks.push({ name, status: "PENDING", detail });

  const close = (a, b, eps = 1e-9) => Math.abs(a - b) < eps;

  const check = (name, fn) => {
    try {
      const outcome = fn();
      if (outcome && outcome.pending) pending(name, outcome.pending);
      else pass(name, outcome === undefined ? "" : String(outcome));
    } catch (error) {
      if (String(error && error.message).includes(TODO_MARKER)) {
        pending(name, "TODO(hw4): not implemented yet.");
      } else {
        fail(name, String(error && error.message ? error.message : error));
      }
    }
  };

  // 1. Item counts on the fixture.
  check("fixture: item counts match the hand-computed values", () => {
    for (const [stock, expected] of Object.entries(example.itemCounts)) {
      const actual = countItem(basketObjects, stock);
      if (actual !== expected) {
        throw new Error(`countItem(${stock}) = ${actual}, expected ${expected}`);
      }
    }
    return "all five item counts correct";
  });

  // 2. lift == 1 case.
  check("fixture: bread -> milk has lift exactly 1", () => {
    const [breadMilk] = example.rules;
    const joint = countPair(basketObjects, "bread", "milk");
    const support = computeSupport(joint, example.n);
    const confidence = computeConfidence(joint, countItem(basketObjects, "bread"));
    const lift = computeLift(confidence, countItem(basketObjects, "milk"), example.n);
    if (!close(support.value, breadMilk.support)) throw new Error("support mismatch");
    if (!close(confidence.value, breadMilk.confidence)) throw new Error("confidence mismatch");
    if (!close(lift.value, breadMilk.lift)) throw new Error(`lift = ${lift.value}, expected 1`);
    return `support=${support.value.toFixed(4)} confidence=${confidence.value.toFixed(4)} lift=${lift.value.toFixed(4)}`;
  });

  // 3. lift > 1 case.
  check("fixture: milk -> jam has lift greater than 1", () => {
    const rule = example.rules[1];
    const joint = countPair(basketObjects, rule.antecedent[0], rule.consequent[0]);
    const confidence = computeConfidence(joint, countItem(basketObjects, rule.antecedent[0]));
    const lift = computeLift(confidence, countItem(basketObjects, rule.consequent[0]), example.n);
    if (!close(lift.value, rule.lift)) throw new Error(`lift = ${lift.value}, expected ${rule.lift}`);
    if (!(lift.value > 1)) throw new Error("expected lift > 1");
    return `lift=${lift.value.toFixed(4)} (> 1)`;
  });

  // 4. lift < 1 case.
  check("fixture: jam -> eggs has lift less than 1", () => {
    const rule = example.rules[2];
    const joint = countPair(basketObjects, rule.antecedent[0], rule.consequent[0]);
    const confidence = computeConfidence(joint, countItem(basketObjects, rule.antecedent[0]));
    const lift = computeLift(confidence, countItem(basketObjects, rule.consequent[0]), example.n);
    if (!close(lift.value, rule.lift)) throw new Error(`lift = ${lift.value}, expected ${rule.lift}`);
    if (!(lift.value < 1 && lift.value > 0)) throw new Error("expected 0 < lift < 1");
    return `lift=${lift.value.toFixed(4)} (0 < lift < 1)`;
  });

  // 5. Reversed confidence differs.
  check("fixture: reversed confidence differs (jam <-> eggs)", () => {
    const forwardJoint = countPair(basketObjects, "jam", "eggs");
    const forward = computeConfidence(forwardJoint, countItem(basketObjects, "jam"));
    const reversed = computeConfidence(forwardJoint, countItem(basketObjects, "eggs"));
    if (close(forward.value, reversed.value)) {
      throw new Error("confidence should differ between jam -> eggs and eggs -> jam");
    }
    if (!close(forward.value, 1 / 3) || !close(reversed.value, 1 / 2)) {
      throw new Error(`unexpected values: ${forward.value} vs ${reversed.value}`);
    }
    return `jam->eggs=${forward.value.toFixed(4)} vs eggs->jam=${reversed.value.toFixed(4)}`;
  });

  // 6. Duplicate items in a raw basket are counted once.
  check("duplicate items in a raw basket are counted once", () => {
    const deduped = dedupeBasket(["milk", "bread", "milk", "bread", "jam"]);
    if (deduped.length !== 3) throw new Error(`expected 3 unique items, got ${deduped.length}`);
    const duplicateBasket = [["milk", "bread", "milk", "bread"], ["milk"]];
    if (countItem(duplicateBasket, "milk") !== 2) {
      throw new Error("countItem must count each basket once, not each row");
    }
    if (countPair(duplicateBasket, "milk", "bread") !== 1) {
      throw new Error("countPair must count each basket once");
    }
    return "dedupeBasket removed repeats; counting is per basket";
  });

  // 7. Empty result set renders an empty-state note.
  check("empty rule set renders an empty-state note", () => {
    const scratch = document.createElement("div");
    renderResults([], DATASET_INDEX, scratch);
    if (!/no rules passed/i.test(scratch.textContent)) {
      throw new Error("expected an empty-state message");
    }
    return "empty state rendered";
  });

  // 8. Invalid thresholds are rejected.
  check("invalid thresholds are rejected", () => {
    if (validateThresholds(0.01, 0.3).ok !== true) throw new Error("valid thresholds rejected");
    if (validateThresholds(Number.NaN, 0.3).ok !== false) throw new Error("NaN support accepted");
    if (validateThresholds(0.01, 1.5).ok !== false) throw new Error("confidence > 1 accepted");
    if (validateThresholds(0, 0.3).ok !== false) throw new Error("zero support accepted");
    return "valid accepted, invalid rejected";
  });

  // 9. Zero-denominator guards.
  check("zero-denominator guards return undefined metrics", () => {
    if (computeConfidence(0, 0).defined !== false) throw new Error("count(A)=0 must be undefined");
    if (computeLift({ value: 0.5, defined: true }, 0, 5).defined !== false) {
      throw new Error("count(B)=0 must be undefined");
    }
    if (computeSupport(0, 0).defined !== false) throw new Error("N=0 must be undefined");
    return "zero denominators return defined=false";
  });

  // 10. Student function: frequent itemsets (pending until implemented).
  check("findFrequentItemsets reproduces the fixture's frequent itemsets", () => {
    const itemsets = findFrequentItemsets(basketObjects, 0.4);
    const pairs = itemsets.filter((set) => set.items.length === 2);
    if (pairs.length === 0) throw new Error("no frequent pairs found at support >= 0.4");
    return `${itemsets.length} itemsets`;
  });

  // 11. Student function: rules (pending until implemented).
  check("generateRules reproduces the fixture's rules", () => {
    const itemsets = findFrequentItemsets(basketObjects, 0.2);
    const rules = generateRules(itemsets, 0.5);
    if (rules.length === 0) throw new Error("no rules found at support >= 0.2, confidence >= 0.5");
    return `${rules.length} rules`;
  });

  const passed = checks.filter((c) => c.status === "PASS").length;
  const failed = checks.filter((c) => c.status === "FAIL").length;
  const pendingCount = checks.filter((c) => c.status === "PENDING").length;

  const lines = [
    `HW4 self-checks — pass ${passed}, fail ${failed}, pending ${pendingCount} (of ${checks.length})`,
    "",
    ...checks.map((c) => `${c.status.padEnd(7)} ${c.name}${c.detail ? ` — ${c.detail}` : ""}`),
  ];
  const text = lines.join("\n");

  const target = logElement || document.getElementById("testLog");
  if (target) target.textContent = text;

  return { passed, failed, pending: pendingCount, checks };
}

// ---------------------------------------------------------------------------
// Wiring
// ---------------------------------------------------------------------------

/**
 * Populate the stock -> description lookup used by the renderers from the
 * dictionary-encoded tables embedded in `window.HW4`.
 *
 * @returns {void}
 */
function primeDescriptions() {
  data.stocks.forEach((stock, i) => {
    if (!descriptionByStock.has(stock)) {
      descriptionByStock.set(stock, data.descriptions[i]);
    }
  });
}

/**
 * Read the two sliders and return the thresholds as fractions in `(0, 1]`.
 *
 * @returns {{minSupport: number, minConfidence: number}}
 */
function readThresholds() {
  const supportInput = document.getElementById("min-support");
  const confidenceInput = document.getElementById("min-confidence");
  return {
    minSupport: Number(supportInput.value) / 100,
    minConfidence: Number(confidenceInput.value) / 100,
  };
}

/** Update the `<output>` readouts next to the two sliders. */
function syncThresholdLabels(changedId = null) {
  const supportInput = document.getElementById("min-support");
  const confidenceInput = document.getElementById("min-confidence");
  const supportOutput = document.getElementById("min-support-value");
  const confidenceOutput = document.getElementById("min-confidence-value");
  if (supportOutput) supportOutput.textContent = `${Number(supportInput.value).toFixed(1)}%`;
  if (confidenceOutput) confidenceOutput.textContent = `${Number(confidenceInput.value).toFixed(0)}%`;
  const supportNumber = document.getElementById("support-number");
  const confidenceNumber = document.getElementById("confidence-number");
  if (supportNumber && (!changedId || changedId === "min-support")) supportNumber.value = supportInput.value;
  if (confidenceNumber && (!changedId || changedId === "min-confidence")) confidenceNumber.value = confidenceInput.value;
  updateThresholdPresentation();
}

function validThresholdNumbers() {
  const validity = ["support-number","confidence-number"].map((id) => {
    const input = document.getElementById(id);
    const valid = !!input && input.value !== "" && input.checkValidity();
    if (input) input.setAttribute("aria-invalid", String(!valid));
    return valid;
  });
  return validity.every(Boolean);
}

function updateThresholdPresentation() {
  const valid = validThresholdNumbers();
  const support = document.getElementById("support-number");
  const confidence = document.getElementById("confidence-number");
  const hint = document.getElementById("support-count-hint");
  if (hint) hint.textContent = valid && DATASET_INDEX
    ? `At this support, a pattern needs at least ${Math.ceil(DATASET_INDEX.n * Number(support.value) / 100).toLocaleString("en-US")} joint baskets.`
    : "Enter valid percentages within the stated ranges and steps.";
  const stale = RULE_VIEW.hasRun && (!valid || !RULE_VIEW.thresholds ||
    Number(support.value)/100 !== RULE_VIEW.thresholds.minSupport ||
    Number(confidence.value)/100 !== RULE_VIEW.thresholds.minConfidence);
  const notice = document.getElementById("stale-notice");
  if (notice) notice.hidden = !stale;
}

function setRunning(running) {
  RULE_VIEW.running = running;
  ["run-rules","min-support","min-confidence","support-number","confidence-number"].forEach((id) => {
    const input = document.getElementById(id);
    if (input) input.disabled = running;
  });
  const button = document.getElementById("run-rules");
  if (button) button.textContent = running ? "Running…" : "Run rules →";
  const panel = document.querySelector(".results-panel");
  if (panel) panel.setAttribute("aria-busy", String(running));
}

/**
 * Read the thresholds, run the student pipeline, and render the results.
 *
 * @returns {void}
 */
async function runPipeline() {
  if (RULE_VIEW.running) return;
  if (!DATASET_INDEX) {
    setStatus("error", "Dataset not ready yet — reload the page.");
    return;
  }
  if (!validThresholdNumbers()) {
    setStatus("error", "Check the percentages: support 0.5–10 in 0.5-point steps; confidence 10–100 in 5-point steps.");
    updateThresholdPresentation();
    const invalid = document.querySelector('.number-wrap input[aria-invalid="true"]');
    if (invalid) invalid.focus();
    return;
  }
  const { minSupport, minConfidence } = readThresholds();
  const validation = validateThresholds(minSupport, minConfidence);
  if (!validation.ok) {
    setStatus("error", validation.errors.join(" "));
    return;
  }
  setRunning(true);
  setStatus("running", "Mining frequent itemsets and evaluating rule directions…");
  try {
    // Allow the real running state to paint before the synchronous local miner.
    // There is no estimated percentage or fabricated intermediate progress.
    await new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve)));
    const itemsets = findFrequentItemsets(TRANSACTIONS, minSupport);
    const rules = generateRules(itemsets, minConfidence);
    RULE_VIEW.thresholds = { minSupport, minConfidence };
    renderResults(rules, DATASET_INDEX);
    const label = document.getElementById("run-label");
    if (label) label.textContent = `Run: support ≥ ${(minSupport*100).toFixed(1)}% · confidence ≥ ${(minConfidence*100).toFixed(0)}%`;
    setStatus(rules.length ? "complete" : "empty", `Done · ${rules.length.toLocaleString("en-US")} rules found at support ≥ ${(minSupport*100).toFixed(1)}% and confidence ≥ ${(minConfidence*100).toFixed(0)}%.`);
  } catch (error) {
    const message = String(error && error.message ? error.message : error);
    setStatus("error", `Run failed: ${message}. Previous successful results, if any, are retained.`);
    if (!RULE_VIEW.hasRun) document.getElementById("results").innerHTML = '<p class="empty-state">The run did not complete. Check the error above and try again.</p>';
  } finally {
    setRunning(false);
    updateThresholdPresentation();
  }
}

/**
 * Wire the controls once the DOM is ready, then build the dataset index from the
 * baskets already decoded from `window.HW4` and render the summary.
 *
 * The data is already in memory (embedded by `transactions.js`), so there is no
 * fetch and no error path beyond the `window.HW4` guard at the top of this file.
 *
 * @returns {void}
 */
function init() {
  renderWorkedExample(tinyWorkedExample());
  syncThresholdLabels();

  const supportInput = document.getElementById("min-support");
  const confidenceInput = document.getElementById("min-confidence");
  if (supportInput) supportInput.addEventListener("input", () => syncThresholdLabels("min-support"));
  if (confidenceInput) confidenceInput.addEventListener("input", () => syncThresholdLabels("min-confidence"));

  for (const [numberId, rangeId] of [["support-number","min-support"],["confidence-number","min-confidence"]]) {
    const number = document.getElementById(numberId);
    number.addEventListener("input", () => {
      if (number.value !== "" && number.checkValidity()) {
        document.getElementById(rangeId).value = number.value;
        syncThresholdLabels(rangeId);
      } else updateThresholdPresentation();
    });
    number.addEventListener("keydown", (event) => {
      if (event.key === "Enter") { event.preventDefault(); runPipeline(); }
    });
  }
  document.getElementById("rule-search").addEventListener("input", (event) => {
    RULE_VIEW.query = event.target.value; RULE_VIEW.page = 0;
    clearRuleSelection(); if (RULE_VIEW.hasRun) renderRulePage();
  });
  document.getElementById("rule-sort").addEventListener("change", (event) => {
    RULE_VIEW.sort = event.target.value; RULE_VIEW.page = 0;
    if (RULE_VIEW.hasRun) renderRulePage();
  });
  document.getElementById("sort-order").addEventListener("click", (event) => {
    RULE_VIEW.descending = !RULE_VIEW.descending; RULE_VIEW.page = 0;
    event.currentTarget.textContent = RULE_VIEW.descending ? "Descending ↓" : "Ascending ↑";
    event.currentTarget.setAttribute("aria-label", RULE_VIEW.descending ? "Sort descending; switch to ascending" : "Sort ascending; switch to descending");
    if (RULE_VIEW.hasRun) renderRulePage();
  });
  for (const [id, increment] of [["page-prev",-1],["page-next",1]]) {
    document.getElementById(id).addEventListener("click", () => {
      RULE_VIEW.page += increment; renderRulePage();
      const button = document.getElementById(id);
      const fallback = document.getElementById(increment > 0 ? "page-prev" : "page-next");
      (button.disabled ? fallback : button).focus({preventScroll:true});
    });
  }

  const runButton = document.getElementById("run-rules");
  if (runButton) runButton.addEventListener("click", runPipeline);

  const testButton = document.getElementById("run-tests");
  if (testButton) testButton.addEventListener("click", () => runTests());

  primeDescriptions();
  N = TRANSACTIONS.length;
  DATASET_INDEX = buildIndex(TRANSACTIONS);
  try {
    renderDatasetSummary(DATASET_INDEX);
  } catch (error) {
    // `renderDatasetSummary` is scaffolding, but guard it defensively so that an
    // unimplemented TODO(hw4) stub can never take the whole page down at load
    // time. The harness's "Run tests" button stays usable regardless.
    const summary = document.getElementById("dataset-summary-body");
    if (summary) {
      summary.innerHTML =
        '<p class="empty-state">Implement the TODO(hw4) functions to activate ' +
        "the dataset summary.</p>";
    }
  }
  updateThresholdPresentation();
  setStatus("ready", `Ready · ${N.toLocaleString("en-US")} baskets loaded. Set your thresholds and run.`);
}

if (typeof document !== "undefined") {
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
}
