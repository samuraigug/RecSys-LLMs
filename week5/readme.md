# HW5 — From Ratings to Recommendations (Matrix Factorization)

**Course**: LLM4Rec, HSE University
**Instructor**: Seungmin Jin (sedzhin@hse.ru)
**Repository**: https://github.com/dryjins/RecSys-LLMs/tree/main/week5
**Task type**: Assignment (individual)

---

## 1. Learning goals

By the end of this assignment you should be able to:

- Turn a sparse rating log into a **user–movie rating matrix** and explain why an
  unobserved rating is **missing data, not a zero**.
- Split ratings **per user, by timestamp**, into an **80% train / 10% validation /
  10% test** protocol that mimics "predict the future from the past".
- Implement **matrix factorization** with **16-dimensional** user and movie
  vectors and a **dot-product** prediction rule.
- Train the factors with **batched tensor operations in TensorFlow.js**, using
  an optimizer (`tf.train.adam` / `tf.train.sgd`) instead of hand-written
  per-rating JavaScript loops.
- Select a checkpoint by **validation RMSE** and report **test RMSE** against a
  **training-set mean-rating baseline**.
- Generate **Top-10 recommendation lists** that exclude movies the user already
  rated in training or validation.
- Evaluate retrieval with **Recall@10**, treating test ratings **>= 4.0** as
  relevant, and report the evaluation **exclusions** that make the number
  interpretable.
- Reason about whether a recommender helps users **discover** new movies or mainly
  **repeats** their existing preferences, and what that means for a streaming
  platform.

---

## 2. Run instructions

Serve the folder over HTTP and open `week5/index.html` in a modern browser:

```bash
cd week5
python3 -m http.server 8000
# then open http://localhost:8000/
```

The page needs an **HTTP server** (because `data.js` uses `fetch()` to read
`u.data` and `u.item`) and **network access** on first load (the browser fetches
the TensorFlow.js CDN script). It also runs on static hosting such as GitHub
Pages, which serves files over HTTP.

Keep `week5/u.data`, `week5/u.item`, `week5/data.js`, `week5/style.css`, and
`week5/script.js` next to `index.html`. Press **Run tests** to execute the
self-checks, then press **Train**,
**Recommend Top-10**, and **Evaluate Recall@10**. The backend that TensorFlow.js
selected is shown in the **TensorFlow.js status** panel (`#tf-status`).

---

## 3. Dataset provenance

- **Source**: MovieLens 100k (GroupLens / University of Minnesota).
  - Dataset page: `https://grouplens.org/datasets/movielens/100k/`
- **Files used** (byte-identical to Week 2 and Week 3, so the **algorithm** is the
  only variable between the assignments):
  - `u.data` — **100,000 ratings**, tab-separated
    `userId | movieId | rating | timestamp`.
  - `u.item` — **1,682 movies**, pipe-separated
    `id | title | release | video | url | unknown flag | 18 named genre flags`.
    There are 19 genre flags in total. The file contains Latin-1 text; the loader
    decodes its bytes explicitly without changing the data file.
- **Shapes in this dataset**: **943 users**, **1,682 movies**, **100,000 ratings**.
  The highest user id is **943** and the highest movie id is **1,682**, so the
  dense rating matrix is `(943 + 1) x (1682 + 1)`.
- **Timestamps**: every rating in `u.data` carries a Unix timestamp; the split in
  `data.js` sorts each user's ratings by ascending timestamp (ties broken by
  `movieId`).
- **Citation** (as required by the source): F. Maxwell Harper and Joseph A.
  Konstan, "The MovieLens Datasets: History and Context", *ACM Transactions on
  Interactive Intelligent Systems*, 2015.

---

## 4. Tensors and GPU acceleration

### 4.1 What TensorFlow.js is

**TensorFlow.js** is a JavaScript library for **tensor computation and automatic
differentiation**: tensors are multi-dimensional arrays, operations over them are
expressed as library calls, and each operation records enough information to
compute gradients automatically. It runs entirely **in the browser** and can use
the **WebGL** backend to execute tensor operations on the **GPU**, falling back
to the pure-JavaScript **CPU** backend when WebGL is unavailable.

`week5/index.html` already loads the runtime from a CDN before the page scripts:

```html
<script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.22.0/dist/tf.min.js"></script>
<script src="data.js"></script>
<script src="script.js"></script>
```

Because this is a plain (classic) script load, TensorFlow.js exposes a global
**`tf`** object that `week5/script.js` can use directly — no module import is
needed.

### 4.2 Use TensorFlow.js (required)

Implement the **matrix-factorization training** and the **batched scoring** with
TensorFlow.js tensor operations rather than hand-written JavaScript loops over
rating arrays. TensorFlow.js is the GPU-accelerated path expected for this
assignment. The six stubs expect these APIs:

| Purpose | API |
|---|---|
| Learnable factor matrices | `tf.variable` |
| Small random initialization | `tf.randomNormal(shape, mean, std)` |
| Gather rows for a batch of user/movie ids | `tf.gather` |
| Score many (user, movie) pairs at once | `tf.matMul` (and element-wise `mul` / `sum`) |
| Loss for a training batch | `tf.losses.meanSquaredError` |
| Optimizer and in-place update | `tf.train.adam` / `tf.train.sgd` + `optimizer.minimize` |
| Top-K over all item scores | `tf.topk` |
| Automatic release of intermediate tensors | `tf.tidy` |

The result is the same algorithm as a handwritten implementation, but each
mini-batch is one tensor operation executed on the GPU where possible.

### 4.3 Practical points

- **Select the backend and report it.** `initTf()` in `script.js` prefers WebGL,
  falls back to CPU, and writes the chosen backend into `#tf-status` via
  `tf.getBackend()`. State the printed backend in your report: if it says `cpu`,
  the GPU path was not used and your timings are host-CPU numbers.
- **Await readiness once.** Call `await tf.ready()` before the first training
  step so the backend is fully initialized (the provided `initTf()` already does
  this).
- **Manage memory.** Wrap per-step work in `tf.tidy` and **dispose** long-lived
  tensors you no longer need. TensorFlow.js tensors live outside the JavaScript
  heap; undisposed tensors leak GPU memory, especially inside an epoch loop.
  Re-initializing the model (repeated **Train** clicks) must also dispose the
  previous `tf.Variable`s.
- **The first CDN load needs network access.** On a machine without network the
  `tf` global is undefined; the page shows a clear message instead of throwing
  (see `initTf()`).

---

## 5. Required student work

### 5.1 Implementation Task

- **Dataset**: MovieLens. Use `userId` and `movieId` to identify users and movies;
  use `title` and `genres` to describe movies. Reuse the same `u.data` / `u.item`
  files as Week 2 and Week 3.
- **Data preparation**: build a user–movie rating matrix. Treat missing ratings as
  **unobserved, not zeros**.
- **Data splitting**: sort each user's ratings by **timestamp** and use
  **80% train / 10% validation / 10% test** (provided by `data.js`).
- **Matrix factorization**: implement JavaScript matrix factorization with
  **16-dimensional** user and movie vectors; the predicted rating is their
  **dot product**.
- **Model evaluation**: select the checkpoint using **validation RMSE**. Compare
  **test RMSE** against a **training-set mean-rating baseline**.
- **Recommendation**: generate **Top-10** lists for **two users**. Show movie
  titles and scores, and **exclude movies rated in training or validation**.
- **Retrieval evaluation**: treat test ratings **>= 4.0** as relevant and report
  **Recall@10** using **training-set movies as candidates**. Report the
  **evaluation exclusions** for unseen ids (relevant test movies absent from the
  training pool) and for users without an eligible relevant test movie.

This submission implements **six model functions** in `week5/script.js`:
`initializeFactors`, `trainMF`, `predictRating`, `selectBestCheckpoint`,
`recommendTopK`, and `recallAtK`.

### 5.2 Business & Algorithmic Analysis (exactly ONE question)

Answer **only** the following question (in the course report, using numbers and
titles produced by your own code):

**Familiar Choices vs. New Discoveries.** Do your recommendations help users
**discover** movies, or do they mainly **repeat** their existing preferences?
Compare two users' Top-10 lists with their past ratings. Use specific movies or
genres to explain what the lists are worth to a streaming platform.

---

## 6. Definitions

Let a rating be `r(u, i)`, the rating user `u` gave movie `i`.

The model stores a **global training mean** `mu` and two **16-dimensional** factor
vectors per id: `p_u` for user `u` and `q_i` for movie `i`. The predicted rating is

```
r_hat(u, i) = mu + p_u . q_i
```

where `.` is the dot product. The result is clamped to `[1, 5]`.

The dense factor matrices are indexed by the **raw** user and movie ids:

- `userFactors`: shape `[numUsers + 1, 16]` = `[944, 16]`, row index = `userId`.
- `itemFactors`: shape `[maxMovieId + 1, 16]` = `[1683, 16]`, row index = `movieId`.
- `globalMean`: the mean rating over the **training** split.

**RMSE** (root-mean-square error) over a split of `N` ratings:

```
RMSE = sqrt( (1 / N) * SUM_over_ratings ( r_hat(u, i) - r(u, i) )^2 )
```

**Mean-rating baseline**: predict the training-set mean `mu` for every pair and
compute the RMSE on the same split. It is the number your factorization must beat.
The provided `computeBaselineRMSE` implements it.

**Recall@K**: with **relevant** = test ratings with `rating >= 4.0` and
**candidates** = the set of movie ids appearing anywhere in the training split,
for each evaluable user

```
recall_u = ( number of that user's relevant candidate movies in the Top-K list ) / ( number of that user's relevant candidate movies )
```

and `Recall@K` is the **mean of `recall_u` over the evaluable users**. Movies
rated in training or validation are excluded from both that user's list and its
eligible relevant set. **Unseen ids** means relevant test movies absent from the
global training pool, not movies already seen by the active user. These reasons
must be distinguished (the supplied MovieLens data has no repeated user–item pairs).

---

## 7. Implementation & analysis tasks

1. Read the six implemented model functions in `week5/script.js`:
   `predictRating`, `initializeFactors`, `trainMF`, `selectBestCheckpoint`,
   `recommendTopK`, and `recallAtK`. The rest of the file (TensorFlow.js
   bootstrap, `computeRMSEFor`, `computeBaselineRMSE`, formatting helpers, all DOM
   renderers, the hyper-parameter reader, the pipeline wrapper, and the test
   harness) is scaffolding and should be left as-is.
2. Run the page, press **Run tests**, and record the baseline counts. With the
   stubs untouched the harness reports **4 passed / 0 failed / 7 pending** (the
   provided helpers pass; the seven stub-backed checks are reported as PENDING
   rather than FAIL). After a correct implementation all **11** checks pass.
3. Implement the six stubs with TensorFlow.js tensor operations, following the
   API guide in §4.2 and the per-stub contract notes in the code.
4. Press **Train** with the default hyper-parameters and report the **validation
   RMSE** at the selected checkpoint and the **test RMSE** of that checkpoint.
   Compare the test RMSE with the **training-set mean-rating baseline** from
   `computeBaselineRMSE(testSet)` and state whether the factorization beats it.
5. Generate the **Top-10** list for **two users** and record each movie's **title**
   and **score**. Confirm that none of the listed movies was rated by that user in
   training or validation.
6. Evaluate **Recall@10** on the test split and report the overall recall together
   with the **exclusions**: the number of unseen relevant test items and the
   number of users without a relevant candidate.
7. Answer the single **§5.2** analysis question, comparing the two users' Top-10
   lists with their past ratings and citing specific movies or genres.
8. Verify every citation you use (paper, URL, authors, year) before submitting,
   and export the unmodified `session.json` for the report (see §8).

---

## 8. Submission instructions

Submit the **modified `week5/` directory** (or your own repository) containing the
files below, plus the course report.

| File | Role |
|---|---|
| `week5/script.js` | implementation of the six model functions |
| `week5/data.js` | data layer (provided, do not modify) |
| `week5/index.html` | page structure (provided, do not modify) |
| `week5/style.css` | styling (provided, do not modify) |
| `week5/readme.md` | this file |

Plus:

- `report.pdf` — IEEE-aligned report per the course homework guidelines
  (`docs/homework-guidelines/guidelines.md`), answering §5.2.
- `session.json` — the **unmodified** export from your AI tool
  (`opencode export <sessionID>`).

- **No Jupyter notebook** (`.ipynb`) is part of this deliverable.
- **No separate memo** is required; the analysis belongs in the report.
- **Late submissions are not accepted.**

---

## 9. Grading criteria

Grading follows the course homework guidelines, **§8 — Rubric Criteria in Detail**
(`docs/homework-guidelines/guidelines.md`). The two criteria are **binary** (0 or
1) and combine into a per-assignment score of **0, 1, or 2**:

- **c1 — Problem & References.** Clear problem statement; all references valid;
  attribution accurate. A hallucinated citation, a broken reference URL, or a
  fabricated author/year is a **Gate 0** failure of c1.
- **c2 — Solution & Evidence.** The solution works (the page runs and the numbers
  shown match the code's own output); the reasoning is accurate; and verification
  is cited (for example a hand-computed RMSE on one pair, a re-run, a source read
  beyond the abstract, or a cross-check of a reported number against the code).

Read **§10** of the same guidelines for the **Gate 0** failure modes that apply
most directly here:

- a **hallucinated citation** (a paper or URL that does not exist);
- **fabricated verification** — claiming you hand-computed or re-ran something you
  did not;
- a **broken solution** — the page does not run, or the numbers shown do not match
  the code's own output;
- a **hallucinated API call** — reporting use of a TensorFlow.js function that
  does not exist or is not actually used.

If you cannot verify a critical output, say so honestly in the report's AI-usage
section rather than claiming verification you did not perform.

---

## Local preparation audit exceptions (2026-10-16)

The original seven files and unmodified harness are preserved under
`../evidence/hw5-audit/original/week5/`. `../PLAN.md` records evidence and scope.
At the end of stage 1 all six model stubs were unimplemented. Confirmed starter issues required
these narrow exceptions to the leave-scaffolding-unchanged instruction:

- Decode `u.item` as Latin-1; keep both dataset files byte-identical.
- Correct only candidate **data** in `REC_FIXTURE` and `recallFixture`: competing
  movies occur in another user's train. Rank-1/rank-11 and train/val assertions
  remain unchanged.
- Release harness-owned fixture variables and temporary tensors. Repeated tests
  must not leak tensors or dispose the application's saved model.
- Correct the non-executable optimizer example (`returnCost=false` returns null)
  and document that `trainMF` must restore best-epoch weights.
- Correct the file-format and seen/unseen explanations above.

Observed stage-1 browser baseline and post-repair results were both **4 PASS / 0 FAIL /
7 PENDING**. This does not validate the unimplemented model.

Stage 2 adds separate `evaluationViews` (known training IDs, candidate pool,
train/validation seen sets, and eligible relevant test sets) without mutating the
raw splits. A labelled baseline table reports eligible validation/test and the
additional full-test baseline with their denominators. Empty evaluations are
shown as `N=0 / not evaluable`, not as perfect RMSE. The supplied parsers, matrix,
split, metric helpers and existing renderers remain unchanged; stage 2 also left
all six model bodies untouched. Bootstrap prepares the views and appends the new summary.

Stage 3 implemented `initializeFactors` and `predictRating`. The factors have
F=16 by default, stdDev=0.05, user seed=2026 and item seed=2027; the TF.js4.22.0
signature is `randomNormal(shape, mean, stdDev, dtype, seed)` (seed is argument5).
A separate shuffle seed=2028 was reserved there and is used by stage-4 `trainMF`.
Only the passed training ratings determine the mean and nonzero factor rows;
ID0 and user/item rows without observations are zero. Reinitialization disposes
only the previous model's variables, and `tf.tidy` releases initial tensors.
Scalar prediction is `clip(mu + dot(p_u,q_i),1,5)` and returns a JavaScript number.
Unknown IDs use the clamped mean, while no model returns the starter sentinel0.
Empty train stores mu0 but its predictions clamp to1, matching the batched helper;
this is not an evaluable trained model. No user/item bias, genre features or cosine
normalization is added. Latent factors are not genres, and scores are not probabilities.

Observed stage-3 in-page results were **6 PASS /0 FAIL /5 PENDING**.

### Method — training and checkpoint selection (stage 4)

`trainMF(train,val,opts)` now runs synchronously, leaving the model's original
variables in place. Each batch contains observed training ratings only: int32
user/item indices, float32 targets, `gather`, element-wise multiply and a sum over
the latent dimensions (16 in the main run). Adam uses beta1=0.9, beta2=0.999 and
epsilon=1e-7, an explicit variable list `[userFactors,itemFactors]`, and
`returnCost=false` (returns null; no `.dispose()` call on it).

The exact optimized objective per batch B is:

```text
MSE_B = (1/|B|) * sum_(u,i,r in B) (mu + dot(p_u,q_i) - r)^2
loss_B = MSE_B + reg * (sum_all_user_factors P^2 + sum_all_item_factors Q^2)
```

The training prediction is **unclipped**. L2 is the **full sum**, neither a mean
over matrix elements nor divided by the batch size. Reported train/validation
RMSE uses predictions clipped to [1,5] and is not this regularized loss.
Mulberry32 seed2028 and Fisher-Yates shuffle only training-row indices; each
epoch starts from the original index order with a continuing seeded RNG.

Validation is filtered to IDs observed in the passed train; raw arrays remain
unchanged and global test targets are never read. Empty train returns []; no
eligible validation causes a clear error instead of using the empty helper's0.
Hyperparameters must be finite with positive integer epochs/batchSize, lr>0 and
reg>=0. The supplied UI reader still normalizes epoch/batch inputs as documented.

Every epoch is retained in history. Independent tensor snapshots of both matrices
are replaced only on strict validation improvement and assigned back before return;
an exact tie keeps the earliest epoch. `selectBestCheckpoint` also resolves ties
by epoch in unsorted history, never mutates it, returns null for empty history,
and reports malformed/non-finite rows explicitly. Optimizer slots, batch tensors
and snapshots are disposed. On optimizer failure, direct trainMF rolls back to
the best finite/initial weights and rethrows, retaining completed history rows.

Observed stage-4 in-page results were **8 PASS /0 FAIL /3 PENDING**. At that point
`recommendTopK` and `recallAtK` remained unimplemented. A real MovieLens engineering
smoke ran3epochs (lr.01, reg.05, batch4096), selected epoch2 on eligible validation,
and saved all history/factors/options/seeds under `../evidence/hw5-audit/`.
This is not the required defaults20epoch or final model-selection experiment;
no test RMSE, Top10 or Recall is reported yet. All checks were executed by the AI
agent and are not the student's confirmed personal verification.

**Narrow controller exception:** the supplied runPipeline kept an old checkpoint
after a later invalid training attempt initialized new weights. The before/after
reproduction is in stage4 UI evidence. Its only changes reset checkpoint/history
before a run and dispose/clear the active model and checkpoint on failure; the
successful synchronous pipeline and supplied renderers/metric helpers are preserved.

### Method — common training-only candidates and macro Recall (stage 5)

All six model functions are now implemented. Both report lists and retrieval
evaluation use the shared `getRecommendationContext`, which reads train,
validation and catalog IDs only (not test targets):

```text
C = distinct movieIds anywhere in train
S_u = movieIds in train_u UNION validation_u
C_u = C minus S_u
R_u = distinct test movieIds with rating >=4.0 in C_u, for a known train user
Recall_u@K = |TopK_u INTERSECT R_u| / |R_u|
Recall@K = mean over users with |R_u|>0 of Recall_u@K
```

This is a macro average, not total hits divided by total relevant items. Test
movies remain candidates when they occur in another user's train. Candidates
are not sampled, limited to test positives, or restricted to the active user's
own train. Valid source data have catalog entries for every rating ID.

`recommendTopK` computes clipped(mu+dot) scores through `tf.matMul`, masks ID0,
non-catalogue IDs, IDs outside C and own train/validation IDs to -Infinity **before**
`tf.topk`, and checks the same mask on the returned indices. The result is sorted
by score DESC then movieId ASC. No epsilon perturbs scores. TF.js4.22.0 guarantees
lower indices first on equal topk values; raw row index is movieId. The boundary
tie was verified on WebGL and CPU. Fewer than K candidates returns all of them;
no model, no candidates, unknown train user, or non-positive/non-integer K gives[].

`recallAtK` returns exactly `{overallRecall,perUser,excluded}`. Relevant items are
sets, so repeated eligible pairs do not inflate a user's denominator.
`excluded.unseenIds` is the unique relevant movie IDs absent from global train;
`usersWithoutRelevant` is the unique test users without eligible relevant items.
Unknown-user rows and own-seen repeated pairs are counted separately in evidence,
not mislabeled as unseen movies. An empty population returns technical0 with no
perUser rows and displays `N=0 / not evaluable`. A missing model with evaluable
users gives a clear diagnostic rather than an unmeasured zero score.

Observed in-page result: **11 PASS /0 FAIL /0 PENDING**. Original assertions are
byte-identical; fixture data are the repaired stage-1 version: REC movies3–5 and
recall competitors4–12 occur in another user's train. Replaying only original
fixture DATA with the completed functions produces **9 PASS /2 FAIL /0 PENDING**
(checks9/10); it is distinct from the untouched original application's4/0/7.
The exact data-only diff and original seven files are preserved in evidence.

An independent plain-JS full sort over handwritten factors, then Python set/dot
arithmetic, confirmed Recall values0,1/2,1, macro.5 vs micro5/7, ties, near ties,
test availability, all masks and separate exclusion reasons. These are synthetic
verification examples, not MovieLens quality results or personal student checks.
The final real lists and Recall are recorded in the stage-6 validation-selected experiment below.

**Narrow retrieval-DOM exception:** old lists/Recall were still displayed after
a new successful or failed Train. Retrieval panels are now cleared at the start
of every run. Obsolete implementation screen hints were replaced with truthful
messages; the empty-population label was made explicit. The renderer calculations,
training/checkpoint code, course assertions and page structure are preserved.

### Experiment and labelled test evaluation (stage 6)

The experiment used one fixed per-user temporal split, F16 and seeds2026/2027/2028.
Users1/200, their positive train/validation genre profiles and train-popularity Q1
(count<=5, inclusive) were frozen before actual20epoch trials and real Top10.
Test composition/baseline and earlier engineering-smoke validation were already
known; this is not a completely blind holdout.

Defaults20/.01/.05/64 were run first. After checkpoint/gradient/full-sum-L2 checks,
validation-only reg.001/.0001/.00001 runs and one justified40epoch extension were
compared. The immutable selection lock chose40epochs/lr.01/reg.0001/batch64,
checkpoint epoch24, validationRMSE0.9930039598, **before** test/Recall.

Primary test (known train IDs, N10732): baseline1.2257193843, defaults1.2257194294,
final1.0435732115 — reduction0.1821461728 or14.860349%. Defaults has no meaningful
improvement. Additional full-test N10785 is labelled separately with mean fallback.
Final macroRecall@10=0.0540415969568 over861users;13unseen relevant movieIds and
82users without eligible relevant. All inputs, histories, factors, full candidate
ranks, independent arithmetic, both actual Top10 and P1–P3 cards are stored in
`../evidence/hw5-audit/stage6/`; `../PLAN.md` is the sole project state file.

**Necessary narrow UI addition:** the original pipeline showed only validation,
not MFtestRMSE. Two new helpers compute the current checkpoint's validation,
primarytest and additionalfulltest metrics once and render a labelled table inside
the existing training panel. Two controller lines store the result and pass that
same object to the renderer; the experiment report uses those exact result objects.
All six algorithm-function hashes remain identical to the pre-test protocol.
This UI addition was made only after the selection lock, so validation search
did not see test outputs. The validation-only verification driver omits automatic
post-fit test display when reproducing the protocol with the final UI.

Actual RunTests remained11/0/0 before/after a full locked40epoch Train repeat;
epoch24, P/Q hashes, test metrics, lists and Recall reproduced. A separately
preregistered real-data memory test used1epoch warmup+3identical1epoch resource
cycles:2live model tensors after training/harness,0after explicitdisposal.
Those resource cycles are not extra model-selection candidates. All checks were
executed by AI; student personal numerical verification is still unconfirmed,
and the final report/package is not yet ready for submission.

*Generated 2026-10-06 from HW5 work order.*

### Final report and reproducibility audit (Codex, outside OpenCode)

The English and Russian reports contain all twenty recommendation rows, history
comparisons, user-specific verdicts and product decisions, the restored epoch-24
checkpoint, cohort-labelled RMSE and macro Recall denominators. The package
README describes replay and the final manifest binds the files by SHA-256.
The final documentation edit removes obsolete stub-completion instructions;
application code and data are byte-identical to the completed stage-6 run.

The native session records the student's self-report of independently running
the scenarios and checking absence of seen films (zero overlap). It does not
record a separately hand-derived RMSE or macro Recall. AI numerical verification
and archive smoke checks are identified separately in evidence.

Final report status: Draft. F07 still requires a separately recorded student
numeric result with identified inputs and method. The general self-report above
is preserved without treating AI calculations as student work.
