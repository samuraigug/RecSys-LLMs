/**
 * week5/script.js — Matrix Factorization with TensorFlow.js (HW5).
 *
 * This is a plain (classic) script, NOT an ES module. `index.html` loads the
 * TensorFlow.js runtime from a CDN before this file, exposing the global `tf`:
 *
 *   <script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.22.0/dist/tf.min.js"></script>
 *   <script src="data.js"></script>
 *   <script src="script.js"></script>
 *
 * `week5/data.js` (loaded before this file) exposes the MovieLens 100k data
 * layer as globals: `movies`, `ratings`, `numUsers`, `numMovies`,
 * `ratingMatrix`, `SPLIT_EXCLUSIONS`, `loadData()` and `splitByTimestamp()`.
 * This file reads those globals, renders a dataset / split summary, and wires
 * the page controls. The factorization itself is built on `tf.Variable`s and
 * tensor ops so it can run on the WebGL (GPU) backend.
 *
 * BACKEND: `initTf()` prefers WebGL, falls back to CPU, and writes the chosen
 * backend into `#tf-status`. If the CDN did not load (`typeof tf ===
 * 'undefined'`) the page shows a clear message and stops instead of throwing.
 *
 * WHAT YOU MUST IMPLEMENT (`TODO(hw5)` — each stub throws until you write it):
 *   1. `predictRating`        — scalar gather + dot + globalMean, clamp [1, 5].
 *   2. `initializeFactors`    — two `tf.Variable`s + the global mean.
 *   3. `trainMF`              — batched tensor training loop with L2 / Adam.
 *   4. `selectBestCheckpoint` — earliest entry with the smallest val RMSE.
 *   5. `recommendTopK`        — `matMul` + `tf.topk`, exclude train/val items.
 *   6. `recallAtK`            — Recall@K against relevant (>= 4.0) test items.
 * Stages 3–4 implement initialization, prediction, training and checkpointing;
 * stage 5 implements recommendation and macro Recall with a shared train-only pool.
 *
 * PROVIDED for you (scaffolding): `initTf`, the shared metric helpers
 * (`computeRMSEFor`, `computeBaselineRMSE`), formatting helpers, every DOM
 * renderer, the hyper-parameter reader, the training pipeline wrapper, and the
 * automated test harness. Leave the provided code as-is and implement only the
 * six stubs.
 *
 * The harness reports unimplemented stubs as PENDING rather than FAIL, so the
 * page stays usable before and after the assignment is done.
 *
 * @module week5/script
 */

// ---------------------------------------------------------------------------
// Typedefs
// ---------------------------------------------------------------------------

/**
 * @typedef {Object} Movie
 * @property {number} id
 * @property {string} title
 * @property {string[]} genres
 */

/**
 * @typedef {Object} Rating
 * @property {number} userId
 * @property {number} movieId
 * @property {number} rating
 * @property {number} timestamp
 */

/**
 * @typedef {Object} Model
 * @property {tf.Variable} userFactors Dense factor matrix `[numUsers + 1, F]`,
 *   row index = raw `userId`.
 * @property {tf.Variable} itemFactors Dense factor matrix `[maxMovieId + 1, F]`,
 *   row index = raw `movieId`.
 * @property {number} globalMean       Mean rating of the training set.
 * @property {Object} [trainingInfo]   Plain run metadata/history; no live optimizer/snapshot tensors.
 */

/**
 * @typedef {Object} HistoryRow
 * @property {number} epoch
 * @property {number} trainRMSE
 * @property {number} valRMSE
 */

/**
 * @typedef {Object} TrainOptions
 * @property {number} epochs    Number of full passes over the training set.
 * @property {number} lr        Learning rate.
 * @property {number} reg       L2 regularization strength.
 * @property {number} batchSize Mini-batch size.
 */

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

/** Number of latent factors used by the default pipeline. */
const NUM_FACTORS = 16;

/** Independent initialization and deterministic batch-shuffle seeds. */
const MF_SEEDS = Object.freeze({ users: 2026, items: 2027, shuffle: 2028 });
const FACTOR_INIT_STD = 0.05;

/** Training split (80% of each user's ratings, oldest first). */
let trainSet = [];

/** Validation split (10% of each user's ratings). */
let valSet = [];

/** Test split (10% of each user's ratings, newest). */
let testSet = [];

/** Derived, training-ID-only views; original split arrays are never filtered in place. */
let evaluationViews = null;

/** Cached ranking IDs/seen sets only; scoring must not inspect test targets. */
let recommendationContext = null;

/**
 * The fitted factorization, or `null` before `initializeFactors` runs.
 *
 * @type {Model|null}
 */
let model = null;

/**
 * Metric history produced by `trainMF`.
 *
 * @type {HistoryRow[]}
 */
let trainingHistory = [];

/**
 * The history entry returned by `selectBestCheckpoint`, or `null`.
 *
 * @type {HistoryRow|null}
 */
let bestCheckpoint = null;

/** Name of the TensorFlow.js backend selected by `initTf` (`"webgl"` / `"cpu"`). */
let tfBackend = null;

// ---------------------------------------------------------------------------
// TensorFlow.js bootstrap (provided)
// ---------------------------------------------------------------------------

/**
 * Select the fastest available TensorFlow.js backend and report it.
 *
 * Tries WebGL (GPU) first and falls back to the pure-JS CPU backend. Writes
 * `backend: <name>` into `#tf-status` when that element exists. If the CDN
 * failed, `tf` is undefined: this renders a readable message and returns
 * `null` instead of letting the page throw an uncaught `ReferenceError`.
 *
 * @returns {Promise<string|null>} the selected backend, or `null` when TF.js is missing.
 */
async function initTf() {
  const statusEl = document.getElementById("tf-status");
  if (typeof tf === "undefined") {
    const message =
      "TensorFlow.js failed to load. Check your network connection (the CDN script) and reload the page.";
    if (statusEl) {
      statusEl.textContent = message;
      statusEl.className = "error";
    } else if (document.body) {
      const banner = document.createElement("p");
      banner.className = "error";
      banner.textContent = message;
      document.body.insertBefore(banner, document.body.firstChild);
    }
    return null;
  }

  try {
    await tf.setBackend("webgl");
  } catch (error) {
    await tf.setBackend("cpu");
  }
  await tf.ready();

  tfBackend = tf.getBackend();
  if (statusEl) statusEl.textContent = `TensorFlow.js backend: ${tfBackend}`;
  return tfBackend;
}

/**
 * True when `candidate` is a TF.js factor model (dense `tf.Variable` rows).
 *
 * Used to route `computeRMSEFor` through the batched tensor path and to keep
 * the fixture-based tests independent of how the model was produced.
 *
 * @param {*} candidate
 * @returns {boolean}
 */
function isTfModel(candidate) {
  return (
    !!candidate &&
    !!candidate.userFactors &&
    typeof candidate.userFactors.dataSync === "function" &&
    !!candidate.itemFactors &&
    typeof candidate.itemFactors.dataSync === "function"
  );
}

// ---------------------------------------------------------------------------
// Formatting helpers (provided)
// ---------------------------------------------------------------------------

/**
 * Format a number to a fixed number of decimals. Non-finite and missing values
 * render as an em dash so the tables never show `NaN` or `undefined`.
 *
 * @param {number|null|undefined} x
 * @param {number} [d=4] decimal places
 * @returns {string}
 */
function fmt(x, d = 4) {
  if (x === null || x === undefined) return "\u2014";
  const n = Number(x);
  if (!Number.isFinite(n)) return "\u2014";
  return n.toFixed(d);
}

/**
 * Look up a movie title by id, with a readable fallback for unknown ids.
 *
 * @param {number|string} id
 * @returns {string}
 */
function movieTitle(id) {
  const numericId = Number(id);
  const movie = movies.find((m) => m.id === numericId);
  return movie ? movie.title : `Movie #${id}`;
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

/**
 * Clamp a number into the inclusive range `[lo, hi]`.
 *
 * @param {number} value
 * @param {number} lo
 * @param {number} hi
 * @returns {number}
 */
function clamp(value, lo, hi) {
  return Math.min(hi, Math.max(lo, value));
}

// ---------------------------------------------------------------------------
// Shared metric helpers (provided)
// ---------------------------------------------------------------------------

/**
 * Root-mean-square error of the current model over a split.
 *
 * Provided so that `trainMF` and the test harness share one implementation.
 *
 * When `model` holds TF.js variables this builds the index tensors ONCE, uses
 * `tf.gather` + element-wise multiply to score every rating in a single batch,
 * and clamps to `[1, 5]` — the same contract as `predictRating`, but fast
 * enough to call every epoch. Out-of-range ids fall back to `globalMean`, and
 * the whole computation runs inside `tf.tidy` so the intermediate tensors are
 * released automatically.
 *
 * When no TF.js model is available it falls back to the scalar
 * `predictRating` loop. That path returns `NaN` instead of throwing while
 * `predictRating` is still a `TODO(hw5)` stub, so the helper never crashes the
 * page or the harness.
 *
 * @param {Rating[]} split
 * @returns {number} `0` for an empty split; `NaN` if the scalar path throws.
 */
function computeRMSEFor(split) {
  if (!split || split.length === 0) return 0;

  if (isTfModel(model)) {
    return tf.tidy(() => {
      const n = split.length;
      const uRows = model.userFactors.shape[0];
      const iRows = model.itemFactors.shape[0];
      const userIds = new Int32Array(n);
      const movieIds = new Int32Array(n);
      const actual = new Float32Array(n);
      const valid = new Float32Array(n);

      for (let k = 0; k < n; k++) {
        const r = split[k];
        userIds[k] = r.userId;
        movieIds[k] = r.movieId;
        actual[k] = r.rating;
        const uOk = r.userId >= 0 && r.userId < uRows;
        const iOk = r.movieId >= 0 && r.movieId < iRows;
        valid[k] = uOk && iOk ? 1 : 0;
      }

      const uIdx = tf.clipByValue(tf.tensor1d(userIds, "int32"), 0, Math.max(0, uRows - 1));
      const iIdx = tf.clipByValue(tf.tensor1d(movieIds, "int32"), 0, Math.max(0, iRows - 1));
      const u = tf.gather(model.userFactors, uIdx);
      const i = tf.gather(model.itemFactors, iIdx);
      const dot = tf.sum(tf.mul(u, i), 1);
      const mask = tf.tensor1d(valid, "float32");
      const pred = tf.clipByValue(dot.mul(mask).add(model.globalMean), 1, 5);
      const diff = pred.sub(tf.tensor1d(actual, "float32"));
      return Math.sqrt(tf.mean(diff.square()).dataSync()[0]);
    });
  }

  let sum = 0;
  try {
    for (const r of split) {
      const diff = predictRating(r.userId, r.movieId) - r.rating;
      sum += diff * diff;
    }
  } catch (error) {
    return Number.NaN;
  }
  return Math.sqrt(sum / split.length);
}

/**
 * Baseline RMSE: predict the training-set mean rating for every pair and
 * measure the error on `split`. This is the number your factorization must beat.
 *
 * @param {Rating[]} split
 * @returns {number} `0` for an empty split (never divides by zero).
 */
function computeBaselineRMSE(split) {
  if (!split || split.length === 0) return 0;
  const mean = trainSet.length
    ? trainSet.reduce((acc, r) => acc + r.rating, 0) / trainSet.length
    : 0;
  let sum = 0;
  for (const r of split) {
    const diff = mean - r.rating;
    sum += diff * diff;
  }
  return Math.sqrt(sum / split.length);
}

// ---------------------------------------------------------------------------
// Evaluation protocol (HW5 stage 2). No factors, training or ranking here.
// The supplied data parsers, split and metric helpers remain unchanged.
// ---------------------------------------------------------------------------

/** A reported empty evaluation is null/not evaluable, not a perfect RMSE of 0. */
function summarizeMeanBaseline(rows, mean, trainingCount) {
  const N = rows.length;
  const evaluable = N > 0 && trainingCount > 0;
  const sse = evaluable
    ? rows.reduce((sum, r) => sum + (mean - r.rating) ** 2, 0)
    : null;
  return { N, evaluable, sse, rmse: evaluable ? Math.sqrt(sse / N) : null };
}

/**
 * Build evaluation views from the three supplied splits only (never ratingMatrix).
 * Raw arrays/rows are read-only inputs; filtered arrays are separate containers.
 * pool and trainUserIds use train only; seenByUser uses train UNION validation.
 * RMSE excludes unknown training IDs, while retrieval also excludes own seen items.
 */
function buildEvaluationViews(train, val, test) {
  const sortedIds = (ids) => Array.from(ids).sort((a, b) => a - b);
  const trainUserIds = new Set(train.map((r) => r.userId));
  const pool = new Set(train.map((r) => r.movieId));
  const mean = train.length ? train.reduce((sum, r) => sum + r.rating, 0) / train.length : 0;
  const seenByUser = new Map();
  for (const split of [train, val]) {
    for (const r of split) {
      if (!seenByUser.has(r.userId)) seenByUser.set(r.userId, new Set());
      seenByUser.get(r.userId).add(r.movieId);
    }
  }

  const rmseView = (raw) => {
    const rows = [];
    const userIds = new Set(), movieIds = new Set();
    let unknownUserRatings = 0, unknownMovieRatings = 0, bothRatings = 0;
    for (const r of raw) {
      const unknownUser = !trainUserIds.has(r.userId), unknownMovie = !pool.has(r.movieId);
      if (unknownUser) { unknownUserRatings++; userIds.add(r.userId); }
      if (unknownMovie) { unknownMovieRatings++; movieIds.add(r.movieId); }
      if (unknownUser && unknownMovie) bothRatings++;
      if (!unknownUser && !unknownMovie) rows.push(r);
    }
    return {
      rawCount: raw.length, rows,
      excluded: { ratings: raw.length - rows.length, unknownUserRatings, unknownMovieRatings,
        bothRatings, userIds: sortedIds(userIds), movieIds: sortedIds(movieIds) },
      baseline: summarizeMeanBaseline(rows, mean, train.length),
    };
  };

  const relevantRows = test.filter((r) => r.rating >= 4.0);
  const eligibleRelevantRows = [], relevantByUser = new Map();
  const unseenIds = new Set(), unknownRelevantUsers = new Set();
  let unseenRelevantRatings = 0, unknownUserRelevantRatings = 0, seenRelevantRatings = 0;
  // Reason counts may overlap; excludedRelevantRatings is the deduplicated union.
  for (const r of relevantRows) {
    const unknownUser = !trainUserIds.has(r.userId), unseen = !pool.has(r.movieId);
    const seen = !!seenByUser.get(r.userId)?.has(r.movieId);
    if (unknownUser) { unknownUserRelevantRatings++; unknownRelevantUsers.add(r.userId); }
    if (unseen) { unseenRelevantRatings++; unseenIds.add(r.movieId); }
    if (seen) seenRelevantRatings++;
    if (unknownUser || unseen || seen) continue;
    eligibleRelevantRows.push(r);
    if (!relevantByUser.has(r.userId)) relevantByUser.set(r.userId, new Set());
    relevantByUser.get(r.userId).add(r.movieId);
  }
  const testUsers = new Set(test.map((r) => r.userId));
  return {
    sources: { train, val, test }, trainUserIds, pool, seenByUser, mean,
    rmse: { train: rmseView(train), val: rmseView(val), test: rmseView(test) },
    baselineFullTest: summarizeMeanBaseline(test, mean, train.length),
    recallPopulation: {
      threshold: 4.0, relevantRows, eligibleRelevantRows, relevantByUser,
      evaluableUsers: sortedIds(relevantByUser.keys()),
      usersWithoutRelevant: sortedIds(Array.from(testUsers).filter((u) => !relevantByUser.has(u))),
      unseenIds: sortedIds(unseenIds), unknownRelevantUsers: sortedIds(unknownRelevantUsers),
      excludedRelevantRatings: relevantRows.length - eligibleRelevantRows.length,
      unseenRelevantRatings, unknownUserRelevantRatings, seenRelevantRatings,
    },
  };
}

/** Splits are immutable after preparation; rebuild if a fixture replaces them. */
function getEvaluationViews() {
  if (!evaluationViews || evaluationViews.sources.train !== trainSet ||
      evaluationViews.sources.val !== valSet || evaluationViews.sources.test !== testSet) {
    evaluationViews = buildEvaluationViews(trainSet, valSet, testSet);
  }
  return evaluationViews;
}

/** Shared candidate protocol for report lists and Recall. Splits/catalog are immutable. */
function getRecommendationContext() {
  if (!recommendationContext || recommendationContext.train !== trainSet ||
      recommendationContext.val !== valSet || recommendationContext.movies !== movies) {
    const pool = new Set(trainSet.map((r) => r.movieId));
    const trainUsers = new Set(trainSet.map((r) => r.userId));
    const seenByUser = new Map();
    for (const split of [trainSet, valSet]) {
      for (const r of split) {
        if (!seenByUser.has(r.userId)) seenByUser.set(r.userId, new Set());
        seenByUser.get(r.userId).add(r.movieId);
      }
    }
    recommendationContext = { train: trainSet, val: valSet, movies, pool, trainUsers, seenByUser,
      catalogIds: new Set(movies.map((m) => m.id)) };
  }
  return recommendationContext;
}

/** Separate labelled output; does not change the supplied split renderer/helper. */
function renderEvaluationSummary(views = getEvaluationViews(), container) {
  const parent = container || document.getElementById("split-summary-body");
  if (!parent) return;
  let target = parent.querySelector(".evaluation-summary");
  if (!target) {
    target = document.createElement("div");
    target.className = "evaluation-summary";
    parent.appendChild(target);
  }
  const row = (label, result) => `<tr><td>${label}</td><td class="num">${result.N}</td>` +
    `<td>${result.evaluable ? "evaluable" : "not evaluable"}</td>` +
    `<td class="num">${result.evaluable ? fmt(result.rmse, 8) :
      (result.N === 0 ? "N=0 / not evaluable" : "not evaluable (no train mean)")}</td></tr>`;
  target.innerHTML = `<h3>Training-only ID evaluation</h3>
    <p>Training mean: ${views.sources.train.length ? fmt(views.mean, 10) : "not evaluable (no train ratings)"}.
    Baseline predicts this one mean for every pair.</p>
    <table class="data-table"><thead><tr><th>Baseline evaluation</th><th>N ratings</th><th>Status</th><th>RMSE</th></tr></thead>
    <tbody>${row("Validation (known train IDs)", views.rmse.val.baseline)}
    ${row("Test (known train IDs; primary)", views.rmse.test.baseline)}
    ${row("Test (all pairs; additional)", views.baselineFullTest)}</tbody></table>
    <p class="hint">Excluded from primary RMSE: validation ${views.rmse.val.excluded.ratings} ratings
    (${views.rmse.val.excluded.movieIds.length} unique movies, ${views.rmse.val.excluded.userIds.length} unique users);
    test ${views.rmse.test.excluded.ratings} ratings (${views.rmse.test.excluded.movieIds.length} unique movies,
    ${views.rmse.test.excluded.userIds.length} unique users). Raw splits are unchanged.</p>
    <p class="hint">Recall population only (not a measured Recall): pool ${views.pool.size} train movies;
    ${views.recallPopulation.eligibleRelevantRows.length} eligible relevant ratings;
    ${views.recallPopulation.evaluableUsers.length} evaluable users;
    ${views.recallPopulation.usersWithoutRelevant.length} users without eligible relevant;
    ${views.recallPopulation.unseenRelevantRatings} unseen relevant ratings /
    ${views.recallPopulation.unseenIds.length} unique movies.</p>`;
}

/** Current restored checkpoint metrics; primary MF/baseline always share warm-ID rows. */
function computeCheckpointMetrics() {
  if (!model || model.trainingInfo?.status !== "trained") return null;
  const views = getEvaluationViews();
  const measure = (rows) => {
    const N = rows.length;
    if (!N) return { N: 0, evaluable: false, baselineRMSE: null, mfRMSE: null, deltaRMSE: null, relativeReduction: null };
    const baselineRMSE = computeBaselineRMSE(rows), mfRMSE = computeRMSEFor(rows);
    return { N, evaluable: true, baselineRMSE, mfRMSE, deltaRMSE: mfRMSE - baselineRMSE,
      absoluteDifference: Math.abs(mfRMSE - baselineRMSE),
      relativeReduction: baselineRMSE ? (baselineRMSE - mfRMSE) / baselineRMSE : null };
  };
  return { selectedEpoch: bestCheckpoint?.epoch ?? model.trainingInfo.selectedCheckpoint.epoch,
    options: { ...model.trainingInfo.options }, seeds: { ...MF_SEEDS }, mean: model.globalMean,
    validation: measure(views.rmse.val.rows), test: measure(views.rmse.test.rows), fullTest: measure(testSet),
    exclusions: { validation: views.rmse.val.excluded, test: views.rmse.test.excluded },
    primaryRule: "Known training user AND movie IDs; same rows for MF and baseline",
    additionalRule: "All test pairs; zero untrained rows/out-of-range IDs give clamped mean fallback" };
}

/** Minimal signed output, using the same result object exported to the experiment report. */
function renderCheckpointMetrics(result) {
  const parent = document.getElementById("training-curve");
  if (!parent || !result) return;
  let target = parent.querySelector(".checkpoint-metrics");
  if (!target) { target = document.createElement("div"); target.className = "checkpoint-metrics"; parent.appendChild(target); }
  const row = (label, r) => `<tr><td>${label}</td><td class="num">${r.N}</td>` +
    `<td class="num">${r.evaluable ? fmt(r.baselineRMSE, 8) : "not evaluable"}</td>` +
    `<td class="num">${r.evaluable ? fmt(r.mfRMSE, 8) : "not evaluable"}</td>` +
    `<td class="num">${r.evaluable ? fmt(r.deltaRMSE, 8) : "N=0"}</td></tr>`;
  target.innerHTML = `<h3>Current checkpoint evaluation — epoch ${result.selectedEpoch}</h3>
    <table class="data-table"><thead><tr><th>Evaluation cohort</th><th>N ratings</th><th>Baseline RMSE</th><th>MF RMSE</th><th>MF − baseline</th></tr></thead>
    <tbody>${row("Validation: known train IDs", result.validation)}
    ${row("Test: known train IDs (primary)", result.test)}
    ${row("Test: all pairs, mean fallback (additional)", result.fullTest)}</tbody></table>
    <p class="hint">Primary MF and baseline use identical rows. Negative MF − baseline means lower RMSE.
    Full-test results have a different denominator and are additional, not interchangeable.</p>`;
}

// ---------------------------------------------------------------------------
// Student model functions (all six implemented after stage 5)
//
// Keep signatures and contracts; stages are implemented separately.
// ---------------------------------------------------------------------------

/**
 * Predict the rating a user would give a movie (scalar).
 *
 * Implemented with TensorFlow.js. The model stores the
 * factors as two dense `tf.Variable`s:
 *   - `model.userFactors` shape `[numUsers + 1, F]`, row index = `userId`
 *   - `model.itemFactors` shape `[maxMovieId + 1, F]`, row index = `movieId`
 *
 * Suggested TF.js path (wrap it in `tf.tidy` and pull out a JS number):
 * ```js
 * const f = model.userFactors.shape[1];
 * return tf.tidy(() => {
 *   const u = model.userFactors.slice([userId, 0], [1, f]); // [1, F]
 *   const i = model.itemFactors.slice([movieId, 0], [1, f]); // [1, F]
 *   const dot = u.mul(i).sum().dataSync()[0];                // scalar
 *   return clamp(model.globalMean + dot, 1, 5);
 * });
 * ```
 * `dataSync()` copies the value to the CPU as a plain number, which survives
 * `tf.tidy` (the tensors do not). Keep this function scalar: `computeRMSEFor`
 * uses its own batched tensor path, and the harness calls it once per pair.
 *
 * Contract:
 *  - Unknown, non-positive or non-integer ids use the clamped global mean.
 *    In-range ids absent from train have zero factor rows, giving the same result.
 *    For empty train, globalMean=0 is stored but predictions clamp to 1, matching
 *    computeRMSEFor; this is not an evaluable trained model.
 *  - Return `0` when the model is not initialized (`model === null`) so callers
 *    can guard cheaply.
 *  - The result is clamped to `[1, 5]`.
 *
 * @param {number} userId
 * @param {number} movieId
 * @returns {number} predicted rating in `[1, 5]`, or `0` without a model
 */
function predictRating(userId, movieId) {
  if (!model) return 0; // Starter's no-model sentinel, not a rating.
  const knownRow = (id, rows) => Number.isInteger(id) && id > 0 && id < rows;
  if (!knownRow(userId, model.userFactors.shape[0]) ||
      !knownRow(movieId, model.itemFactors.shape[0])) {
    return clamp(model.globalMean, 1, 5);
  }
  return tf.tidy(() => {
    const F = model.userFactors.shape[1];
    const user = model.userFactors.slice([userId, 0], [1, F]);
    const item = model.itemFactors.slice([movieId, 0], [1, F]);
    const dot = user.mul(item).sum().dataSync()[0];
    return clamp(model.globalMean + dot, 1, 5);
  });
}

/**
 * Allocate the latent factor variables and the global mean.
 *
 * `globalMean` uses only the passed train array (0 for empty train).
 * TF.js 4.22.0: randomNormal(shape, mean, stdDev, dtype, seed).
 * stdDev=0.05, user seed=2026, item seed=2027. Broadcast row masks leave ID0
 * and rows without train observations exactly zero, including catalog-only IDs.
 * Initial tensors/masks are tidied; only two trainable variables survive.
 *
 * Contract:
 *  - `numFactors` defaults to the module-level `NUM_FACTORS`.
 *  - If a previous `model` exists, dispose its variables (`tf.dispose`) before
 *    replacing it so repeated "Train" clicks do not leak GPU memory.
 *  - Return the `model` object.
 *
 * @param {Rating[]} train
 * @param {number} [numFactors=NUM_FACTORS]
 * @returns {Model}
 */
function initializeFactors(train, numFactors = NUM_FACTORS) {
  if (!Number.isInteger(numFactors) || numFactors <= 0) {
    throw new Error("numFactors must be a positive integer.");
  }
  const globalMean = train.length
    ? train.reduce((sum, r) => sum + r.rating, 0) / train.length
    : 0;
  const userRows = numUsers + 1;
  const itemRows = movies.reduce((max, m) => Math.max(max, m.id), 0) + 1;
  const userMask = new Float32Array(userRows);
  const itemMask = new Float32Array(itemRows);
  for (const r of train) {
    if (Number.isInteger(r.userId) && r.userId > 0 && r.userId < userRows) userMask[r.userId] = 1;
    if (Number.isInteger(r.movieId) && r.movieId > 0 && r.movieId < itemRows) itemMask[r.movieId] = 1;
  }

  // Dispose only this model's variables, never globally registered variables.
  if (model) tf.dispose([model.userFactors, model.itemFactors]);
  model = null;
  let userFactors, itemFactors;
  try {
    model = tf.tidy(() => {
      const users = tf.randomNormal([userRows, numFactors], 0, FACTOR_INIT_STD, "float32", MF_SEEDS.users)
        .mul(tf.tensor2d(userMask, [userRows, 1]));
      const items = tf.randomNormal([itemRows, numFactors], 0, FACTOR_INIT_STD, "float32", MF_SEEDS.items)
        .mul(tf.tensor2d(itemMask, [itemRows, 1]));
      userFactors = tf.variable(users);
      itemFactors = tf.variable(items);
      return { userFactors, itemFactors, globalMean };
    });
  } catch (error) {
    // tf.tidy releases tensors on failure, but tf.Variable needs explicit disposal.
    tf.dispose([userFactors, itemFactors].filter(Boolean));
    throw error;
  }
  return model;
}

/**
 * Train the factorization with batched gradient descent.
 *
 * Implemented synchronously with observed mini-batches and the TF.js optimizer.
 * Score each batch with tensor ops (never one `tf` call per rating) and let the
 * optimizer update both variables in place:
 * ```js
 * const optimizer = tf.train.adam(lr);
 * for (let epoch = 1; epoch <= epochs; epoch++) {
 *   for (const batch of batches) {
 *     optimizer.minimize(() => tf.tidy(() => {
 *       const u = tf.gather(model.userFactors, batch.userIdx);
 *       const i = tf.gather(model.itemFactors, batch.itemIdx);
 *       const pred = tf.sum(u.mul(i), 1).add(model.globalMean);
 *       const mse = tf.losses.meanSquaredError(batch.targets, pred);
 *       const l2 = model.userFactors.square().sum()
 *         .add(model.itemFactors.square().sum()).mul(reg);
 *       return mse.add(l2);
 *     }), false, [model.userFactors, model.itemFactors]);
 *   }
 *   trainingHistory.push({
 *     epoch,
 *     trainRMSE: computeRMSEFor(train),
 *     valRMSE: computeRMSEFor(eligibleVal),
 *   });
 * }
 * return trainingHistory;
 * ```
 * Key TF.js APIs to use: `tf.train.adam(lr)` (or `tf.train.sgd`),
 * `optimizer.minimize(fn, returnCost, varList)`, `tf.gather`,
 * `tf.losses.meanSquaredError`, and `tf.tidy` for per-batch intermediates.
 * With returnCost=false, `minimize` returns null: do not call .dispose() on it.
 * With returnCost=true, dispose the returned scalar cost tensor. Dispose the
 * optimizer and batch tensors when finished as well.
 * Save independent copies of both factor matrices at the best validation epoch,
 * restore those weights with assign() before returning, then dispose snapshots.
 * Selecting a history row in runPipeline does not restore model weights.
 * Actual loss: MSE_batch(mu + dot BEFORE clipping) + reg*(sum(P^2)+sum(Q^2)).
 * L2 is a full-matrix SUM on every batch, not a mean. RMSE clips predictions.
 * Validation uses only IDs present in the passed train. Nonempty train with no
 * eligible validation rows fails explicitly (N=0 is not a perfect checkpoint).
 * model.trainingInfo stores options, seeds, denominators, all history and the
 * selected/restored epoch; it contains no live optimizer or snapshot tensors.
 *
 * Contract:
 *  - `opts = { epochs, lr, reg, batchSize }`. Use mini-batches of `batchSize`
 *    (or full-batch when `batchSize >= train.length`).
 *  - Update `model` in place. Do NOT reallocate it.
 *  - Append one `HistoryRow` per epoch to the module-level `trainingHistory`
 *    and return it: `{ epoch, trainRMSE: computeRMSEFor(train),
 *    valRMSE: computeRMSEFor(eligibleVal) }`. `epoch` is 1-based.
 *  - Shuffling the batch order each epoch is allowed and usually helps.
 *  - Guard `train.length === 0` by returning an empty history.
 *
 * @param {Rating[]} train
 * @param {Rating[]} val
 * @param {TrainOptions} opts
 * @returns {HistoryRow[]}
 */
function trainMF(train, val, opts) {
  if (train.length === 0) {
    trainingHistory = [];
    return trainingHistory;
  }
  const { epochs, lr, reg, batchSize } = opts || {};
  if (!Number.isInteger(epochs) || epochs < 1) throw new Error("epochs must be a finite integer >= 1.");
  if (!Number.isFinite(lr) || lr <= 0) throw new Error("lr must be finite and > 0.");
  if (!Number.isFinite(reg) || reg < 0) throw new Error("reg must be finite and >= 0.");
  if (!Number.isInteger(batchSize) || batchSize < 1) throw new Error("batchSize must be a finite integer >= 1.");
  if (!isTfModel(model) || !(model.userFactors instanceof tf.Variable) ||
      !(model.itemFactors instanceof tf.Variable) || !Number.isFinite(model.globalMean)) {
    throw new Error("Initialize a finite TF.js factor model before training.");
  }
  const P = model.userFactors, Q = model.itemFactors;
  if (P.rank !== 2 || Q.rank !== 2 || P.shape[1] !== Q.shape[1]) {
    throw new Error("User and item factors must be rank-2 matrices with the same width.");
  }
  const finiteFactors = () => tf.tidy(() =>
    tf.logicalAnd(tf.isFinite(P).all(), tf.isFinite(Q).all()).dataSync()[0] === 1);
  if (!finiteFactors()) throw new Error("Factor matrices must contain only finite values.");
  for (const r of train) {
    if (!Number.isInteger(r.userId) || r.userId <= 0 || r.userId >= P.shape[0] ||
        !Number.isInteger(r.movieId) || r.movieId <= 0 || r.movieId >= Q.shape[0] ||
        !Number.isFinite(r.rating) || r.rating < 1 || r.rating > 5) {
      throw new Error("Training ratings must have allocated positive integer IDs and finite ratings in [1,5].");
    }
  }
  // Never inspect the global testSet (even for preparing the validation view).
  const validationView = buildEvaluationViews(train, val, []).rmse.val;
  const eligibleVal = validationView.rows;
  if (eligibleVal.length === 0) throw new Error("No eligible validation ratings; cannot select a checkpoint (N=0).");
  if (!eligibleVal.every((r) => Number.isFinite(r.rating) && r.rating >= 1 && r.rating <= 5 &&
      Number.isInteger(r.userId) && r.userId > 0 && r.userId < P.shape[0] &&
      Number.isInteger(r.movieId) && r.movieId > 0 && r.movieId < Q.shape[0])) {
    throw new Error("Eligible validation ratings must have valid IDs and finite ratings in [1,5].");
  }

  trainingHistory = [];
  const config = { epochs, lr, reg, batchSize };
  const info = {
    status: "training", options: config, seeds: { ...MF_SEEDS }, factors: P.shape[1],
    optimizer: { name: "Adam", beta1: 0.9, beta2: 0.999, epsilon: 1e-7 },
    objective: "MSE_batch(mu + dot, unclipped) + reg * (sum(P^2) + sum(Q^2))",
    metric: "RMSE of clip(mu + dot,1,5); not the regularized objective",
    shuffle: "Mulberry32 + Fisher-Yates; reset index order each epoch; continuous seeded PRNG",
    trainN: train.length, validationN: eligibleVal.length,
    validationExcluded: validationView.excluded,
    batchesPerEpoch: Math.ceil(train.length / batchSize),
  };
  model.trainingInfo = info;
  const order = Uint32Array.from({ length: train.length }, (_, i) => i);
  let state = MF_SEEDS.shuffle >>> 0;
  const random = () => {
    state = (state + 0x6D2B79F5) >>> 0;
    let value = Math.imul(state ^ (state >>> 15), state | 1);
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
    return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
  };
  let optimizer, snapshotP, snapshotQ, best = null;
  try {
    // Initial copies are also a rollback point if training fails before epoch 1.
    snapshotP = P.clone();
    snapshotQ = Q.clone();
    optimizer = tf.train.adam(lr, 0.9, 0.999, 1e-7);
    for (let epoch = 1; epoch <= epochs; epoch++) {
      for (let k = 0; k < order.length; k++) order[k] = k;
      for (let k = order.length - 1; k > 0; k--) {
        const j = Math.floor(random() * (k + 1));
        [order[k], order[j]] = [order[j], order[k]];
      }
      for (let start = 0; start < train.length; start += batchSize) {
        const size = Math.min(batchSize, train.length - start);
        const users = new Int32Array(size), items = new Int32Array(size), targets = new Float32Array(size);
        for (let k = 0; k < size; k++) {
          const r = train[order[start + k]];
          users[k] = r.userId; items[k] = r.movieId; targets[k] = r.rating;
        }
        tf.tidy(() => {
          const userIdx = tf.tensor1d(users, "int32"), itemIdx = tf.tensor1d(items, "int32");
          const labels = tf.tensor1d(targets, "float32");
          optimizer.minimize(() => tf.tidy(() => {
            const pred = tf.gather(P, userIdx).mul(tf.gather(Q, itemIdx)).sum(1).add(model.globalMean);
            const mse = tf.losses.meanSquaredError(labels, pred);
            const penalty = P.square().sum().add(Q.square().sum()).mul(reg);
            return mse.add(penalty);
          }), false, [P, Q]); // returnCost=false returns null; never call .dispose() on it.
        });
      }
      if (!finiteFactors()) throw new Error(`Non-finite factors after epoch ${epoch}; reduce lr/check inputs.`);
      const row = { epoch, trainRMSE: computeRMSEFor(train), valRMSE: computeRMSEFor(eligibleVal) };
      if (!Number.isFinite(row.trainRMSE) || !Number.isFinite(row.valRMSE)) {
        throw new Error(`Non-finite RMSE at epoch ${epoch}.`);
      }
      trainingHistory.push(row);
      if (!best || row.valRMSE < best.valRMSE) {
        const next = tf.tidy(() => ({ users: P.clone(), items: Q.clone() }));
        tf.dispose([snapshotP, snapshotQ]);
        snapshotP = next.users; snapshotQ = next.items;
        best = row; // Strict improvement keeps the earliest epoch on an exact tie.
      }
    }
    P.assign(snapshotP);
    Q.assign(snapshotQ);
    Object.assign(info, { status: "trained", selectedCheckpoint: { ...best },
      lastEpoch: { ...trainingHistory[trainingHistory.length - 1] },
      restoredBestWeights: true, history: trainingHistory.map((row) => ({ ...row })) });
    return trainingHistory;
  } catch (error) {
    // Do not leave diverged weights active; preserve the best finite (or initial) state.
    if (snapshotP && snapshotQ) { P.assign(snapshotP); Q.assign(snapshotQ); }
    Object.assign(info, { status: "failed", error: String(error.message || error),
      restoredBestWeights: !!best, selectedCheckpoint: best ? { ...best } : null,
      history: trainingHistory.map((row) => ({ ...row })) });
    throw error;
  } finally {
    if (optimizer) optimizer.dispose();
    tf.dispose([snapshotP, snapshotQ].filter(Boolean));
  }
}

/**
 * Pick the best epoch from a training history.
 *
 * Return the entry with the smallest finite `valRMSE`. On a tie return the
 * entry with the EARLIEST `epoch` (do not simply take the last minimum). This
 * one is plain JavaScript — no TF.js needed.
 *
 * Contract:
 *  - Return `null` for an empty or missing history.
 *  - Do not mutate the input array.
 *  - Fail explicitly on any malformed row/non-finite or negative RMSE instead of
 *    silently skipping a corrupt history. Epochs must be positive integers.
 *
 * @param {HistoryRow[]} history
 * @returns {HistoryRow|null}
 */
function selectBestCheckpoint(history) {
  if (history == null) return null;
  if (!Array.isArray(history)) throw new Error("history must be an array.");
  let best = null;
  for (const row of history) {
    if (!row || !Number.isInteger(row.epoch) || row.epoch < 1 ||
        !Number.isFinite(row.valRMSE) || row.valRMSE < 0 ||
        !Number.isFinite(row.trainRMSE) || row.trainRMSE < 0) {
      throw new Error("Invalid checkpoint row: positive integer epoch and finite non-negative trainRMSE/valRMSE required.");
    }
    if (!best || row.valRMSE < best.valRMSE ||
        (row.valRMSE === best.valRMSE && row.epoch < best.epoch)) best = row;
  }
  return best;
}

/**
 * Recommend the top-K movies for a user.
 *
 * Score every candidate movie and return the K highest-scoring
 * movies the user has NOT already rated in `trainSet` or `valSet`. Test-set
 * ratings are allowed as candidates. **Prefer the tensor path**:
 * ```js
 * const userRow = model.userFactors.slice([userId, 0], [1, F]); // [1, F]
 * const scores = model.itemFactors.matMul(userRow, false, true)  // [rows, 1]
 *   .squeeze().add(model.globalMean);                            // [rows]
 * // mask excluded ids to a large negative value, then:
 * const { values, indices } = tf.topk(maskedScores, Math.min(K, rows));
 * ```
 * Because `itemFactors` is indexed by raw `movieId`, the `tf.topk` indices ARE
 * the movie ids — no reverse mapping needed. Excluded rows must be masked (for
 * example multiply by a 0/1 mask and add `(1 - mask) * -1e9`); re-apply the
 * same mask when mapping results back so masked ids are dropped.
 *
 * Contract:
 *  - Return `[{ movieId, title, score }, ...]` sorted by descending `score`,
 *    truncated to `K` (fewer if not enough candidates).
 *  - Exclude any movie id that is absent from the global `movies` list, and any
 *    id the user rated in `trainSet` or `valSet`.
 *  - Return `[]` when the model is not initialized, when `K <= 0`, or when the
 *    user has no eligible candidates — never throw.
 *  - `title` comes from `movieTitle(movieId)`.
 *  - Candidate IDs must be positive, allocated, catalogued and present in global
 *    train. Unknown train users and invalid/non-integer K return [].
 *  - Scores are clipped(mu + dot). Equal scores use movieId ASC, with no epsilon.
 *    TF.js4.22.0 topk guarantees lower indices first on ties, including the K boundary.
 *
 * @param {number} userId
 * @param {number} [K=10]
 * @returns {Array<{movieId: number, title: string, score: number}>}
 */
function recommendTopK(userId, K = 10) {
  if (!model || !Number.isInteger(K) || K <= 0 || !Number.isInteger(userId) ||
      userId <= 0 || userId >= model.userFactors.shape[0]) return [];
  const context = getRecommendationContext();
  if (!context.trainUsers.has(userId)) return [];
  const rows = model.itemFactors.shape[0];
  const seen = context.seenByUser.get(userId);
  const mask = new Uint8Array(rows);
  let candidateCount = 0;
  for (const id of context.pool) {
    if (Number.isInteger(id) && id > 0 && id < rows && context.catalogIds.has(id) && !seen?.has(id)) {
      mask[id] = 1;
      candidateCount++;
    }
  }
  if (candidateCount === 0) return [];
  return tf.tidy(() => {
    const user = model.userFactors.slice([userId, 0], [1, model.userFactors.shape[1]]);
    const scores = tf.matMul(model.itemFactors, user, false, true).reshape([rows])
      .add(model.globalMean).clipByValue(1, 5);
    const masked = tf.where(tf.tensor1d(mask, "bool"), scores, tf.fill([rows], Number.NEGATIVE_INFINITY));
    const { values, indices } = tf.topk(masked, Math.min(K, candidateCount), true);
    const ids = indices.dataSync(), topScores = values.dataSync();
    const recs = [];
    for (let k = 0; k < ids.length; k++) {
      const id = ids[k];
      if (mask[id] !== 1) continue; // Re-check the same mask when leaving tensor space.
      recs.push({ movieId: id, title: movieTitle(id), score: topScores[k] });
    }
    return recs.sort((a, b) => b.score - a.score || a.movieId - b.movieId);
  });
}

/**
 * Compute Recall@K for the current model.
 *
 * Retrieval evaluation built on `recommendTopK` with the same cached pool/seen sets.
 *   - relevant = test ratings with `rating >= 4.0`.
 *   - Candidate items = the set of movie ids that appear anywhere in `trainSet`
 *     (a global candidate pool, NOT just the user's own training items).
 *   - For every user with at least one relevant test movie among the
 *     candidates, take `recommendTopK(userId, K)` and count how many of that
 *     user's relevant candidate movies appear in it.
 *
 * Contract:
 *  - Return
 *    `{ overallRecall, perUser, excluded }` where:
 *      `overallRecall` = mean over reported users of `hit / relevant`
 *        (`0` when there are no reported users — never divide by zero);
 *      `perUser` = `[{ userId, relevant, hit }, ...]` (reported users only);
 *      `excluded` = `{ unseenIds, usersWithoutRelevant }`.
 *  - `unseenIds` = distinct relevant test movie ids that are NOT present in
 *    `trainSet` (they can never be recommended from a training-only pool).
 *  - `usersWithoutRelevant` = distinct users that appear in `testSet` but have
 *    no relevant test movie among the candidates.
 *  - Report exclusions explicitly so the recall number is interpretable.
 *  - Relevant items are sets: duplicates do not inflate the denominator.
 *    Own train/validation movies and unknown train users are ineligible.
 *  - If relevant users exist but no model is initialized, fail explicitly rather
 *    than report an unmeasured zero. Empty populations still return technical 0.
 *
 * @param {number} [K=10]
 * @returns {{overallRecall: number, perUser: Array<{userId: number, relevant: number, hit: number}>, excluded: {unseenIds: number[], usersWithoutRelevant: number[]}}}
 */
function recallAtK(K = 10) {
  const context = getRecommendationContext();
  const testUsers = new Set(), unseenIds = new Set(), relevantByUser = new Map();
  for (const r of testSet) {
    testUsers.add(r.userId);
    if (!(r.rating >= 4.0)) continue;
    if (!context.pool.has(r.movieId)) unseenIds.add(r.movieId);
    if (!context.trainUsers.has(r.userId) || !context.pool.has(r.movieId) ||
        context.seenByUser.get(r.userId)?.has(r.movieId)) continue;
    if (!relevantByUser.has(r.userId)) relevantByUser.set(r.userId, new Set());
    relevantByUser.get(r.userId).add(r.movieId);
  }
  const ids = Array.from(relevantByUser.keys()).sort((a, b) => a - b);
  const excluded = {
    unseenIds: Array.from(unseenIds).sort((a, b) => a - b),
    usersWithoutRelevant: Array.from(testUsers).filter((u) => !relevantByUser.has(u)).sort((a, b) => a - b),
  };
  if (ids.length === 0) return { overallRecall: 0, perUser: [], excluded };
  if (!model) throw new Error("Train the model before evaluating Recall.");
  const perUser = ids.map((userId) => {
    const relevant = relevantByUser.get(userId);
    const top = new Set(recommendTopK(userId, K).map((r) => r.movieId));
    let hit = 0;
    for (const id of relevant) if (top.has(id)) hit++;
    return { userId, relevant: relevant.size, hit };
  });
  const overallRecall = perUser.reduce((sum, u) => sum + u.hit / u.relevant, 0) / perUser.length;
  return { overallRecall, perUser, excluded };
}

// ---------------------------------------------------------------------------
// UI rendering (provided)
// ---------------------------------------------------------------------------

/**
 * Render the dataset shape summary.
 *
 * @param {HTMLElement|null} [container]
 * @returns {void}
 */
function renderDatasetSummary(container) {
  const target = container || document.getElementById("dataset-summary-body");
  if (!target) return;
  const maxMovieId = movies.reduce((max, m) => Math.max(max, m.id), 0);
  target.innerHTML = `
    <dl class="summary-list">
      <dt>Movies</dt><dd>${movies.length.toLocaleString("en-US")}</dd>
      <dt>Ratings</dt><dd>${ratings.length.toLocaleString("en-US")}</dd>
      <dt>Users</dt><dd>${numUsers.toLocaleString("en-US")}</dd>
      <dt>Rating matrix</dt><dd>${(numUsers + 1).toLocaleString("en-US")} &times; ${(maxMovieId + 1).toLocaleString("en-US")}</dd>
    </dl>`;
}

/**
 * Render the chronological 80/10/10 split summary.
 *
 * @param {HTMLElement|null} [container]
 * @returns {void}
 */
function renderSplitSummary(container) {
  const target = container || document.getElementById("split-summary-body");
  if (!target) return;
  const total = trainSet.length + valSet.length + testSet.length;
  const pct = (n) => (total ? ((n / total) * 100).toFixed(1) : "0.0");
  const exclusions = typeof SPLIT_EXCLUSIONS === "undefined" ? [] : SPLIT_EXCLUSIONS;
  target.innerHTML = `
    <table class="data-table">
      <thead><tr><th scope="col">Split</th><th scope="col">Ratings</th><th scope="col">Share</th></tr></thead>
      <tbody>
        <tr><td>Train</td><td class="num">${trainSet.length}</td><td class="num">${pct(trainSet.length)}%</td></tr>
        <tr><td>Validation</td><td class="num">${valSet.length}</td><td class="num">${pct(valSet.length)}%</td></tr>
        <tr><td>Test</td><td class="num">${testSet.length}</td><td class="num">${pct(testSet.length)}%</td></tr>
      </tbody>
    </table>
    <p class="hint">${exclusions.length} user(s) skipped while splitting; baseline test RMSE ${fmt(computeBaselineRMSE(testSet))}.</p>`;
}

/**
 * Render the training history as a plain table (no SVG, no chart library).
 *
 * @param {HTMLElement|null} [container]
 * @returns {void}
 */
function renderTrainingCurve(container) {
  const target = container || document.getElementById("training-curve");
  if (!target) return;
  if (!trainingHistory || trainingHistory.length === 0) {
    target.innerHTML =
      '<p class="empty-state">No training history yet. Press &ldquo;Train&rdquo; to fit the model.</p>';
    return;
  }
  let bestIndex = 0;
  for (let i = 1; i < trainingHistory.length; i++) {
    if (trainingHistory[i].valRMSE < trainingHistory[bestIndex].valRMSE) bestIndex = i;
  }
  const rows = trainingHistory
    .map(
      (row, i) => `
      <tr${i === bestIndex ? ' class="best-epoch"' : ""}>
        <td class="num">${row.epoch}</td>
        <td class="num">${fmt(row.trainRMSE)}</td>
        <td class="num">${fmt(row.valRMSE)}</td>
      </tr>`,
    )
    .join("");
  target.innerHTML = `
    <table class="data-table">
      <thead><tr><th scope="col">Epoch</th><th scope="col">Train RMSE</th><th scope="col">Val RMSE</th></tr></thead>
      <tbody>${rows}</tbody>
    </table>`;
}

/**
 * Render the top-K recommendations for a user.
 *
 * @param {number} userId
 * @param {HTMLElement|null} [container]
 * @returns {void}
 */
function renderTopK(userId, container) {
  const target = container || document.getElementById("recommendations");
  if (!target) return;
  if (!model) {
    target.innerHTML = '<p class="empty-state">Train the model before requesting recommendations.</p>';
    return;
  }
  try {
    const recs = recommendTopK(Number(userId), 10);
    if (!recs || recs.length === 0) {
      target.innerHTML = '<p class="empty-state">No eligible recommendations for this user.</p>';
      return;
    }
    target.innerHTML =
      '<ol class="recommendation-list">' +
      recs
        .map(
          (r) =>
            `<li><span class="rec-title">${escapeHtml(r.title)}</span> <span class="rec-score">${fmt(r.score)}</span></li>`,
        )
        .join("") +
      "</ol>";
  } catch (error) {
    target.innerHTML = `<p class="empty-state">Recommendations unavailable. <code>${escapeHtml(error.message)}</code></p>`;
  }
}

/**
 * Render the Recall@K report, including explicit exclusions.
 *
 * @param {number} [K=10]
 * @param {HTMLElement|null} [container]
 * @returns {void}
 */
function renderRecall(K = 10, container) {
  const target = container || document.getElementById("recall-results");
  if (!target) return;
  try {
    const result = recallAtK(K);
    const rows = result.perUser
      .map(
        (u) => `
        <tr>
          <td>${u.userId}</td>
          <td class="num">${u.relevant}</td>
          <td class="num">${u.hit}</td>
          <td class="num">${u.relevant ? fmt(u.hit / u.relevant) : "\u2014"}</td>
        </tr>`,
      )
      .join("");
    target.innerHTML = `
      <p><strong>Recall@${K}:</strong> <span class="num">${result.perUser.length ? fmt(result.overallRecall) : "N=0 / not evaluable"}</span></p>
      <table class="data-table">
        <thead><tr><th scope="col">User</th><th scope="col">Relevant</th><th scope="col">Hits</th><th scope="col">Recall</th></tr></thead>
        <tbody>${rows || '<tr><td colspan="4">No evaluable users.</td></tr>'}</tbody>
      </table>
      <p class="hint">Excluded: ${result.excluded.unseenIds.length} unseen relevant test item(s); ${result.excluded.usersWithoutRelevant.length} user(s) without a relevant candidate.</p>`;
  } catch (error) {
    target.innerHTML = `<p class="empty-state">Recall unavailable. <code>${escapeHtml(error.message)}</code></p>`;
  }
}

/**
 * Read the training hyper-parameters from the DOM, falling back to defaults when
 * the inputs are absent or invalid.
 *
 * @returns {TrainOptions}
 */
function readHyperparameters() {
  const readNumber = (id, fallback) => {
    const el = document.getElementById(id);
    if (!el) return fallback;
    const value = Number(el.value);
    return Number.isFinite(value) ? value : fallback;
  };
  return {
    epochs: Math.max(1, Math.round(readNumber("epochs", 20))),
    lr: readNumber("lr", 0.01),
    reg: readNumber("reg", 0.05),
    batchSize: Math.max(1, Math.round(readNumber("batch-size", 64))),
  };
}

/**
 * Populate the user `<select>` from the training split.
 *
 * @returns {void}
 */
function populateUserSelect() {
  const select = document.getElementById("user-select");
  if (!select) return;
  const ids = Array.from(new Set(trainSet.map((r) => r.userId))).sort((a, b) => a - b);
  select.innerHTML = ids.length
    ? ids.map((id) => `<option value="${id}">User ${id}</option>`).join("")
    : '<option value="">No users</option>';
}

/** Never leave previous-model retrieval results visible during/after another Train. */
function resetRetrievalOutputs() {
  const recs = document.getElementById("recommendations");
  const recall = document.getElementById("recall-results");
  if (recs) recs.innerHTML = '<p class="empty-state">No recommendations for this run yet. After training, press &ldquo;Recommend Top-10&rdquo;.</p>';
  if (recall) recall.innerHTML = '<p class="empty-state">Recall not evaluated for this run. After training, press &ldquo;Evaluate Recall@10&rdquo;.</p>';
}

/**
 * Run the full training pipeline: initialize, train, select the best checkpoint,
 * and refresh the training curve.
 *
 * @returns {void}
 */
function runPipeline() {
  const status = document.getElementById("status");
  if (!movies.length || trainSet.length === 0) {
    if (status) status.textContent = "Data not loaded yet — reload the page.";
    return;
  }
  const opts = readHyperparameters();
  try {
    bestCheckpoint = null;
    trainingHistory = [];
    resetRetrievalOutputs();
    if (status) status.textContent = "Training\u2026";
    initializeFactors(trainSet, NUM_FACTORS);
    trainingHistory = trainMF(trainSet, valSet, opts);
    bestCheckpoint = selectBestCheckpoint(trainingHistory);
    renderTrainingCurve();
    model.evaluationResults = computeCheckpointMetrics();
    renderCheckpointMetrics(model.evaluationResults);
    if (status) {
      status.textContent = bestCheckpoint
        ? `Trained ${opts.epochs} epoch(s) on ${tfBackend || "tf"} — best val RMSE ${fmt(bestCheckpoint.valRMSE)} at epoch ${bestCheckpoint.epoch}.`
        : `Trained ${opts.epochs} epoch(s) — no history recorded.`;
    }
  } catch (error) {
    if (model) tf.dispose([model.userFactors, model.itemFactors]);
    model = null;
    bestCheckpoint = null;
    const message = String(error && error.message ? error.message : error);
    if (status) status.textContent = message;
    const target = document.getElementById("training-curve");
    if (target) {
      target.innerHTML = `<p class="empty-state">Training failed. Check the inputs and retry &ldquo;Train&rdquo;. <code>${escapeHtml(message)}</code></p>`;
    }
  }
}

/**
 * Wire the controls once the data has been loaded. Rendering is defensive so an
 * unimplemented stub can never take the whole page down at load time.
 *
 * @returns {void}
 */
function init() {
  populateUserSelect();
  renderDatasetSummary();
  renderSplitSummary();
  renderTrainingCurve();

  const testButton = document.getElementById("run-tests");
  if (testButton) testButton.addEventListener("click", () => runTests());

  const trainButton = document.getElementById("train");
  if (trainButton) trainButton.addEventListener("click", runPipeline);

  const recommendButton = document.getElementById("recommend-btn");
  if (recommendButton) {
    recommendButton.addEventListener("click", () => {
      const select = document.getElementById("user-select");
      renderTopK(select ? select.value : 1);
    });
  }

  const recallButton = document.getElementById("recall-btn");
  if (recallButton) recallButton.addEventListener("click", () => renderRecall(10));

  const status = document.getElementById("status");
  if (status) {
    status.textContent = `Dataset ready: ${movies.length.toLocaleString("en-US")} movies, ${trainSet.length.toLocaleString("en-US")} train ratings on ${tfBackend || "tf"}. Press \u201cTrain\u201d to fit the model.`;
  }
}

// ---------------------------------------------------------------------------
// Test harness
// ---------------------------------------------------------------------------

/** Marker carried by every unimplemented stub error message. */
const TODO_MARKER = "TODO(hw5)";

/** A five-rating toy training set used by several checks. */
const TOY_RATINGS = [
  { userId: 1, movieId: 1, rating: 5, timestamp: 1 },
  { userId: 1, movieId: 2, rating: 4, timestamp: 2 },
  { userId: 1, movieId: 3, rating: 3, timestamp: 3 },
  { userId: 2, movieId: 1, rating: 2, timestamp: 1 },
  { userId: 2, movieId: 2, rating: 5, timestamp: 2 },
];

/** Movies matching `TOY_RATINGS`. */
const TOY_MOVIES = [
  { id: 1, title: "Toy One", genres: [] },
  { id: 2, title: "Toy Two", genres: [] },
  { id: 3, title: "Toy Three", genres: [] },
];

/** A tiny train/val/test fixture for the recommendation checks. */
const REC_FIXTURE = {
  movies: [
    { id: 1, title: "Alpha", genres: [] },
    { id: 2, title: "Bravo", genres: [] },
    { id: 3, title: "Charlie", genres: [] },
    { id: 4, title: "Delta", genres: [] },
    { id: 5, title: "Echo", genres: [] },
  ],
  train: [
    { userId: 1, movieId: 1, rating: 5, timestamp: 1 },
    // Candidates must occur in global train, without being seen by user 1.
    { userId: 2, movieId: 3, rating: 5, timestamp: 1 },
    { userId: 2, movieId: 4, rating: 4, timestamp: 2 },
    { userId: 2, movieId: 5, rating: 3, timestamp: 3 },
  ],
  val: [{ userId: 1, movieId: 2, rating: 4, timestamp: 2 }],
  test: [{ userId: 1, movieId: 3, rating: 5, timestamp: 3 }],
};

/**
 * Build a TF.js factor model from sparse score maps.
 *
 * `userScores[id]` / `itemScores[id]` may be a number (written into factor 0,
 * so the dot product is just the product) or an array (written across factors).
 * Rows are dense and indexed by raw id, exactly like the real model, so the
 * recommendation / recall fixtures exercise the same tensor path as training.
 *
 * @param {Object<number, number|number[]>} userScores
 * @param {Object<number, number|number[]>} itemScores
 * @param {number} [globalMean=0]
 * @returns {Model}
 */
function fixtureModel(userScores, itemScores, globalMean = 0) {
  const F = NUM_FACTORS;
  const userIds = Object.keys(userScores).map(Number);
  const itemIds = Object.keys(itemScores).map(Number);

  // A number goes into factor 0 (so the dot product is the product); an array
  // is copied across the first factors.
  const rowsFor = (value) => {
    if (Array.isArray(value) || ArrayBuffer.isView(value)) return Array.from(value);
    const row = new Array(F).fill(0);
    row[0] = Number(value);
    return row;
  };

  const maxUser = Math.max(numUsers, ...userIds);
  const maxMovieFromMovies = movies.reduce((m, x) => Math.max(m, x.id), 0);
  const maxItem = Math.max(maxMovieFromMovies, ...itemIds);

  const uData = new Float32Array((maxUser + 1) * F);
  for (const id of userIds) {
    const row = rowsFor(userScores[id]);
    for (let f = 0; f < Math.min(F, row.length); f++) uData[id * F + f] = row[f];
  }
  const iData = new Float32Array((maxItem + 1) * F);
  for (const id of itemIds) {
    const row = rowsFor(itemScores[id]);
    for (let f = 0; f < Math.min(F, row.length); f++) iData[id * F + f] = row[f];
  }

  return tf.tidy(() => ({
    userFactors: tf.variable(tf.tensor2d(uData, [maxUser + 1, F])),
    itemFactors: tf.variable(tf.tensor2d(iData, [maxItem + 1, F])),
    globalMean,
  }));
}

/**
 * A recall fixture with 12 movies, one user whose single relevant test movie
 * (movie 2) is placed at a chosen rank by the hand-made model.
 *
 * @param {number} targetRank 1 to put movie 2 first, 11 to put it last.
 * @returns {{movies: Movie[], train: Rating[], val: Rating[], test: Rating[], model: Model}}
 */
function recallFixture(targetRank) {
  const movies = [];
  for (let id = 1; id <= 12; id++) {
    movies.push({ id, title: `Movie ${id}`, genres: [] });
  }
  const train = [
    { userId: 1, movieId: 1, rating: 5, timestamp: 1 },
    { userId: 2, movieId: 2, rating: 5, timestamp: 1 },
    { userId: 2, movieId: 3, rating: 4, timestamp: 2 },
  ];
  // Ten competitors for movie 2 must belong to the training-only pool.
  for (let id = 4; id <= 12; id++) {
    train.push({ userId: 2, movieId: id, rating: 3, timestamp: id });
  }
  const val = [];
  const test = [{ userId: 1, movieId: 2, rating: 5, timestamp: 2 }];
  const itemScores = { 1: 3.0 };
  if (targetRank === 1) {
    itemScores[2] = 5.0;
    for (let id = 3; id <= 12; id++) itemScores[id] = 1.0 + 0.01 * id;
  } else {
    itemScores[2] = 1.0;
    for (let id = 3; id <= 12; id++) itemScores[id] = 1.1 + 0.01 * id;
  }
  return { movies, train, val, test, model: fixtureModel({ 1: 1 }, itemScores) };
}

/**
 * Run `fn` with the module-level data globals temporarily replaced by a fixture,
 * then restore them. Keeps the checks independent of the loaded dataset.
 *
 * @param {{movies?: Movie[], train?: Rating[], val?: Rating[], test?: Rating[], model?: Model|null, history?: HistoryRow[]}} fixture
 * @param {() => any} fn
 * @returns {any}
 */
function withFixture(fixture, fn) {
  const saved = { movies, trainSet, valSet, testSet, model, trainingHistory };
  try {
    movies = fixture.movies !== undefined ? fixture.movies : movies;
    trainSet = fixture.train || [];
    valSet = fixture.val || [];
    testSet = fixture.test || [];
    model = fixture.model !== undefined ? fixture.model : null;
    trainingHistory = fixture.history || [];
    // Release temporary slices created by checks; variables need explicit disposal.
    return tf.tidy(fn);
  } finally {
    if (model && model !== saved.model) tf.dispose(model);
    movies = saved.movies;
    trainSet = saved.trainSet;
    valSet = saved.valSet;
    testSet = saved.testSet;
    model = saved.model;
    trainingHistory = saved.trainingHistory;
  }
}

/**
 * Run the automated self-checks. Checks that call an unimplemented
 * `TODO(hw5)` stub are reported as PENDING rather than FAIL, so the harness is
 * useful before and after the assignment is implemented.
 *
 * @param {HTMLElement|null} [logElement] element that receives the text log
 * @returns {{passed: number, failed: number, pending: number, checks: Array<Object>}}
 */
function runTests(logElement) {
  const checks = [];
  const close = (a, b, eps = 1e-9) => Math.abs(a - b) < eps;

  const pass = (name, detail) => checks.push({ name, status: "PASS", detail });
  const fail = (name, detail) => checks.push({ name, status: "FAIL", detail });
  const pending = (name, detail) => checks.push({ name, status: "PENDING", detail });

  const check = (name, fn) => {
    try {
      const outcome = fn();
      if (outcome && outcome.pending) pending(name, outcome.pending);
      else pass(name, outcome === undefined ? "" : String(outcome));
    } catch (error) {
      if (String(error && error.message).includes(TODO_MARKER)) {
        pending(name, "TODO(hw5): not implemented yet.");
      } else {
        fail(name, String(error && error.message ? error.message : error));
      }
    }
  };

  // 1. Shared helper: baseline RMSE on a toy split.
  check("computeBaselineRMSE: toy split matches the hand-computed value", () =>
    withFixture(
      {
        train: [
          { userId: 1, movieId: 1, rating: 2, timestamp: 1 },
          { userId: 1, movieId: 2, rating: 4, timestamp: 2 },
        ],
      },
      () => {
        const rmse = computeBaselineRMSE(trainSet);
        if (!close(rmse, 1)) throw new Error(`rmse = ${rmse}, expected 1`);
        return `rmse=${fmt(rmse)}`;
      },
    ),
  );

  // 2. Shared helper: zero-denominator guard.
  check("computeBaselineRMSE: empty split returns 0 (no division by zero)", () =>
    withFixture({ train: [{ userId: 1, movieId: 1, rating: 3, timestamp: 1 }] }, () => {
      if (computeBaselineRMSE([]) !== 0) throw new Error("expected 0");
      return "guarded";
    }),
  );

  // 3. Formatting helper.
  check("fmt: formats numbers and guards non-finite input", () => {
    if (fmt(1.23456) !== "1.2346") throw new Error(`fmt(1.23456) = ${fmt(1.23456)}`);
    if (fmt(2, 2) !== "2.00") throw new Error(`fmt(2, 2) = ${fmt(2, 2)}`);
    if (fmt(Number.NaN) !== "\u2014") throw new Error("NaN must render as an em dash");
    return "ok";
  });

  // 4. Formatting helper.
  check("movieTitle: resolves ids and falls back for unknown ids", () =>
    withFixture({ movies: [{ id: 7, title: "Seven Samurai", genres: [] }] }, () => {
      if (movieTitle(7) !== "Seven Samurai") throw new Error("known id failed");
      if (movieTitle(999) !== "Movie #999") throw new Error(`fallback = ${movieTitle(999)}`);
      return "ok";
    }),
  );

  // 5. Student stub: prediction and clamp (TF.js gather + dot).
  check("predictRating: clamps to [1, 5] and falls back to the global mean", () =>
    withFixture(
      {
        movies: TOY_MOVIES,
        train: TOY_RATINGS,
        model: fixtureModel({ 1: [1, 2] }, { 3: [3, 4] }, 3),
      },
      () => {
        const pred = predictRating(1, 3);
        if (pred !== 5) throw new Error(`predictRating = ${pred}, expected 5 (3 + 11 clamped)`);
        const fallback = predictRating(100000, 3);
        if (fallback !== 3) throw new Error(`out-of-range user must return globalMean, got ${fallback}`);
        return `clamped=${pred}, fallback=${fallback}`;
      },
    ),
  );

  // 6. Student stub: TF.js variable allocation on the toy 5-rating set.
  check("initializeFactors: allocates tf.Variable factors and the global mean", () =>
    withFixture({ movies: TOY_MOVIES, train: TOY_RATINGS, model: null }, () => {
      const created = initializeFactors(TOY_RATINGS, 4);
      if (!created || !model) throw new Error("model was not assigned");
      if (!close(model.globalMean, 3.8)) {
        throw new Error(`globalMean = ${model.globalMean}, expected 3.8`);
      }
      if (!isTfModel(model)) throw new Error("userFactors/itemFactors are not tf tensors");
      if (model.userFactors.shape[1] !== 4 || model.itemFactors.shape[1] !== 4) {
        throw new Error(
          `factor width = ${model.userFactors.shape[1]} / ${model.itemFactors.shape[1]}, expected 4`,
        );
      }
      if (model.itemFactors.shape[0] !== 4) {
        throw new Error(`item rows = ${model.itemFactors.shape[0]}, expected 4 (max movie id 3 + 1)`);
      }
      if (model.userFactors.shape[0] < 3) {
        throw new Error(`user rows = ${model.userFactors.shape[0]}, expected at least 3`);
      }
      const row = model.userFactors.slice([1, 0], [1, 4]).dataSync();
      if (!Array.from(row).every(Number.isFinite)) {
        throw new Error("user factor row is not finite");
      }
      return `user [${model.userFactors.shape}], item [${model.itemFactors.shape}]`;
    }),
  );

  // 7. Student stub: batched training loop.
  check("trainMF: returns one finite history row per epoch", () =>
    withFixture({ movies: TOY_MOVIES, train: TOY_RATINGS, val: TOY_RATINGS, model: null }, () => {
      initializeFactors(TOY_RATINGS, 4);
      const history = trainMF(TOY_RATINGS, TOY_RATINGS, {
        epochs: 3,
        lr: 0.02,
        reg: 0.05,
        batchSize: 2,
      });
      if (!Array.isArray(history) || history.length !== 3) {
        throw new Error(`history length = ${history && history.length}, expected 3`);
      }
      history.forEach((row, i) => {
        if (row.epoch !== i + 1 && row.epoch !== i) {
          throw new Error(`row ${i} epoch = ${row.epoch}`);
        }
        if (!Number.isFinite(row.trainRMSE) || !Number.isFinite(row.valRMSE)) {
          throw new Error(`row ${i} RMSE is not finite`);
        }
      });
      return history.map((r) => `e${r.epoch}:${fmt(r.trainRMSE)}`).join(" ");
    }),
  );

  // 8. Student stub: checkpoint selection and tie-break.
  check("selectBestCheckpoint: smallest valRMSE wins, ties keep the earliest epoch", () => {
    const history = [
      { epoch: 2, trainRMSE: 0.9, valRMSE: 0.4 },
      { epoch: 1, trainRMSE: 1.0, valRMSE: 0.3 },
      { epoch: 3, trainRMSE: 0.8, valRMSE: 0.3 },
    ];
    const best = selectBestCheckpoint(history);
    if (!best || best.epoch !== 1) {
      throw new Error(`best epoch = ${best && best.epoch}, expected 1 (tie -> earliest)`);
    }
    return `best epoch ${best.epoch}`;
  });

  // 9. Student stub: recommendation filtering (tensor matMul + topk).
  check("recommendTopK: excludes train+val items but allows test-rated items", () =>
    withFixture(
      {
        movies: REC_FIXTURE.movies,
        train: REC_FIXTURE.train,
        val: REC_FIXTURE.val,
        test: REC_FIXTURE.test,
        model: fixtureModel({ 1: 1 }, { 1: 5, 2: 4, 3: 3, 4: 2, 5: 1 }),
      },
      () => {
        const recs = recommendTopK(1, 10);
        const ids = recs.map((r) => r.movieId);
        if (ids.includes(1)) throw new Error("movie 1 is in train and must be excluded");
        if (ids.includes(2)) throw new Error("movie 2 is in val and must be excluded");
        if (!ids.includes(3)) throw new Error("movie 3 is test-rated and must be eligible");
        for (let i = 1; i < recs.length; i++) {
          if (recs[i - 1].score < recs[i].score) {
            throw new Error("results are not sorted by descending score");
          }
        }
        return ids.join(", ");
      },
    ),
  );

  // 10. Student stub: recall at rank 1 vs rank 11.
  check("recallAtK: relevant movie at rank 1 scores 1.0; at rank 11 scores 0.0", () => {
    const rankOne = withFixture({ ...recallFixture(1), history: [] }, () => recallAtK(10));
    if (!close(rankOne.overallRecall, 1)) {
      throw new Error(`rank-1 recall = ${rankOne.overallRecall}, expected 1`);
    }
    const rankEleven = withFixture({ ...recallFixture(11), history: [] }, () => recallAtK(10));
    if (!close(rankEleven.overallRecall, 0)) {
      throw new Error(`rank-11 recall = ${rankEleven.overallRecall}, expected 0`);
    }
    return `rank1=${fmt(rankOne.overallRecall)} rank11=${fmt(rankEleven.overallRecall)}`;
  });

  // 11. Student stub: empty-result guard.
  check("recallAtK: empty test set returns 0 without dividing by zero", () =>
    withFixture(
      {
        movies: REC_FIXTURE.movies,
        train: REC_FIXTURE.train,
        val: [],
        test: [],
        model: fixtureModel({ 1: 1 }, { 1: 1, 2: 1, 3: 1, 4: 1, 5: 1 }),
      },
      () => {
        const result = recallAtK(10);
        if (result.overallRecall !== 0) {
          throw new Error(`expected 0, got ${result.overallRecall}`);
        }
        if (!Array.isArray(result.perUser) || result.perUser.length !== 0) {
          throw new Error("perUser must be empty");
        }
        if (!result.excluded) throw new Error("excluded report missing");
        return "guarded";
      },
    ),
  );

  const passed = checks.filter((c) => c.status === "PASS").length;
  const failed = checks.filter((c) => c.status === "FAIL").length;
  const pendingCount = checks.filter((c) => c.status === "PENDING").length;

  const lines = [
    `HW5 self-checks — pass ${passed}, fail ${failed}, pending ${pendingCount} (of ${checks.length})`,
    "",
    ...checks.map((c) => `${c.status.padEnd(7)} ${c.name}${c.detail ? ` \u2014 ${c.detail}` : ""}`),
  ];
  const text = lines.join("\n");

  const target = logElement || document.getElementById("testLog");
  if (target) target.textContent = text;

  return { passed, failed, pending: pendingCount, checks };
}

// ---------------------------------------------------------------------------
// Bootstrap
// ---------------------------------------------------------------------------

/**
 * Select the TF.js backend, load the MovieLens data, split it chronologically,
 * and wire the page. `data.js` must be loaded before this script.
 *
 * If TensorFlow.js is missing, `initTf` renders a clear message and this
 * function stops without throwing.
 *
 * @returns {Promise<void>}
 */
window.onload = async function () {
  const status = document.getElementById("status");
  const backend = await initTf();
  if (!backend) return;

  try {
    if (status) status.textContent = "Loading data\u2026";
    await loadData();
    const split = splitByTimestamp();
    trainSet = split.train;
    valSet = split.val;
    testSet = split.test;
    getEvaluationViews();
    init();
    renderEvaluationSummary();
  } catch (error) {
    console.error("Initialization failed:", error);
    if (status) status.textContent = `Error: ${error.message}`;
  }
};
