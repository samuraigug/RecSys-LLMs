# Week 2: baseline and verification

This is a chronological audit log. Statements that the branch was uncommitted describe the checkpoint when each check was recorded; use Git history for the final publication status.

Working copy: `avetisyan_gurgen_a02`; branch `codex/a02-movie-recommender`, created from the teacher's `main` at `b27cd3f65bde29d07a11dff682373461348ed9ae`. Before code changes, `git status --short --branch` printed only `## main...origin/main`, and `git diff --stat` was empty. No files from HW1 were copied or edited.

## Baseline (local HTTP server + headless Chrome, before code changes)

- 1682 movie options plus the placeholder; empty selection: `Please select a movie first.`
- `Toy Story (1995)` (id 1) parsed as `Children's, Comedy, Crime` and recommended `Aladdin and the King of Thieves (1996), Aladdin (1992)`.
- `Unforgiven (1992)` (id 203) parsed with zero genres and recommended `Toy Story (1995), GoldenEye (1995)`.
- `unknown` (id 267) parsed as `Action`.
- `Star Wars (1977)` (id 50) recommended `Return of the Jedi (1983), Empire Strikes Back, The (1980)`.
- Film 543 appeared as `Mis�rables, Les (1995)`.
- Original `script.js` used intersection/union (Jaccard), excluded only the seed ID, then took `.slice(0, 2)`.

## Dataset audit (before code changes)

- All 1682 `u.item` lines have exactly 24 pipe-delimited fields and exactly 19 binary genre flags. Flag 0 is `unknown`; flags 1–18 correspond to the 18 named genres. No malformed rows.
- 13 movies, including `Unforgiven` (203), have only the Western flag. Two movies, `unknown` (267) and `Good Morning` (1373), have only the `unknown` flag and **no known genre**.
- UTF-8 decoding of the original `u.item` bytes produces 12 replacement characters; decoding the same bytes as ISO-8859-1 produces zero replacements. For example, id 543 becomes `Misérables, Les (1995)`; id 1633 becomes `Á köldum klaka (Cold Fever) (1994)`.

## After changes

- `node --test week2/recommender.test.js` (from repository root): 3/3 pass. Tests compare the actual parser with all 1682 raw records, confirm Latin-1 accents, and independently check the cosine denominator and zero-vector case.
- A local HTTP server served the unchanged HTML and updated JavaScript to headless Chrome. The browser loaded 1682 movies and 100000 ratings; the empty selection still requested a film. For each of ids 1, 50, 203, 543, 246, and 1633, browser assertions found exactly five distinct numbered titles and none matching the seed; ids 267 and 1373 returned `no known genres`. There were no JavaScript page errors.
- Genre changes: Toy Story (1): `Children's, Comedy, Crime` -> `Animation, Children's, Comedy`; Unforgiven (203): `[]` -> `Western`; unknown (267): `Action` -> `[]`; Star Wars (50): `Adventure, Animation, Sci-Fi, Thriller, Western` -> `Action, Adventure, Romance, Sci-Fi, War`.
- The local browser now shows `Misérables, Les (1995)` for id 543 and `Á köldum klaka (Cold Fever) (1994)` for id 1633 without replacement characters.
- Manual cosine check: Toy Story has 3 known genres (Animation, Children's, Comedy), Aladdin (1992) has those three plus Musical (4). Their dot product is 3, their norms are `sqrt(3)` and `sqrt(4)`, hence cosine `3 / sqrt(12) = 0.8660254037844387` (the browser returned the same value). Baseline Jaccard for the same pair was `3 / 4 = 0.75`.
- Browser Top-5 for Toy Story: `Aladdin and the King of Thieves`, `Aladdin`, `Goofy Movie, A`, `Santa Clause, The`, `Home Alone`. For Unforgiven, the five now start with `Tombstone`, `Wild Bill`, `Wyatt Earp`; previously the two displayed were `Toy Story`, `GoldenEye`.
- `git diff --check` succeeded.

## Profile mode and watched-film audit

Before this extension, `git status --short --branch` confirmed the local `codex/a02-movie-recommender` branch still had the uncommitted `data.js`, `script.js`, `style.css` changes and this evidence file and test file. `index.html` was unchanged from the teacher's main. No HW1 branch or `session.json` was used.

The page now has separate one-film and profile Top-5 panels, plus an optional MovieLens user selector. With no user, watched means the films manually selected in either panel. With a user, watched means **all** film IDs in that user's `u.data` history (any rating), plus any manually selected seeds; same-title duplicates are excluded too. The selectable positive seeds for a MovieLens user are restricted to ratings 4–5. The profile is the coordinate-wise mean of at least three selected binary genre vectors. A diagnostic lists the genre weights and flags the case when no genre is shared by all seeds (a heuristic warning, not proof of bad recommendations).

`node --test week2/recommender.test.js`: 5/5 pass. New checks cover an actual three-film average, all-rated exclusion for user 1, a 2-star film matching a 4-star seed, and rejection of a low-rated or two-film profile.

Local HTTP server + headless Chrome, assertions on visible results and dataset:

- Loaded 1682 films and 943 users. Selecting fewer than 3 profile films produced a validation message.
- Manual seeds Toy Story (1), Star Wars (50), Unforgiven (203): nine nonzero profile coordinates, each `1/3`, no genre shared by all; warning displayed. Both modes returned five distinct titles, excluding the three seeds. Single-film Toy Story Top-5: Aladdin and the King of Thieves, Aladdin, Goofy Movie, Santa Clause, Home Alone. Profile Top-5: Return of the Jedi, Empire Strikes Back, Transformers: The Movie, Kid in King Arthur's Court, Operation Dumbo Drop.
- User 1: `u.data` contains 272 rated/watched movies; 163 positive (rating 4–5) options were selectable. Angels and Insects (20) is a 4-star seed; Mad Love (36) has rating 2, cannot be selected as liked, and does not appear in recommendations. Profile seeds Star Wars (50), Empire Strikes Back (172), Return of the Jedi (181), all rated 5. Both visible Top-5 had 5 distinct films and **zero titles or IDs in the 272-film watched set**. Profile Top-5: African Queen, Judge Dredd, Star Trek: Generations, Star Trek: The Motion Picture, Star Trek V: The Final Frontier. No JavaScript page errors.

Final recheck after input validation: manual Toy Story + Aladdin + Unforgiven shows Animation `0.67`, Western `0.33` in the profile; browser again returned five distinct films. With user 1, single-film Star Wars and the three-film Star Wars / Empire / Jedi profile each returned five distinct films with **Top-5 ∩ watched = ∅** (272 watched movie IDs); no page errors. The test suite also explicitly rejects a low-rated seed even if passed outside the normal dropdown.

The site is still a genre-only recommender: choosing several very different seeds can dilute each preference. The no-common-genre warning exposes this limitation but does not claim to resolve it. Popularity, fatigue, retention, and historical time-split evaluation are outside this implementation.

## Independent final audit (latest local working tree)

This section supersedes the earlier test totals and statements about which diagnostics the page provides. The working tree is still on `codex/a02-movie-recommender`, based on upstream `b27cd3f`; the changes are local and uncommitted.

| Check | Observed result | Evidence / method |
| --- | --- | --- |
| `u.item` integrity and genre mapping | 1682 unique IDs; every row has 24 pipe fields and 19 binary flags; 18 known genre columns match the parser for **every row**. Two items have only the `unknown` flag, 13 have only Western. | Independent raw-byte audit; `node --test week2/recommender.test.js` checks every parsed genre vector against the raw flags. UTF-8 would corrupt 12 characters; Latin-1 restores `Misérables`. |
| `u.data` integrity | 100000 unique (user, item) pairs, 943 users, 1682 referenced items; no missing item IDs, out-of-range ratings, noninteger/negative timestamps or malformed rows. Rating frequency for 1–5: 6110, 11370, 27145, 34174, 21201. | Independent four-field/tab audit plus full-row verification in the Node suite. Malformed, duplicate, empty and orphan rows are now rejected instead of being silently dropped from watched history. |
| Formula / empty genre vector | Toy Story vs Aladdin cosine `3 / sqrt(12) = 0.8660254038` (not Jaccard `0.75`); zero norm returns 0. | Hand calculation and Node test; in Chrome selecting `unknown` shows `no known genres`, and the two unknown-only films are absent from the profile picker. |
| Empty selection / ordinary Top-5 | Both empty buttons show explicit selection errors. With Star Wars alone, its Top-5 is Jedi, Empire, Starship Troopers, African Queen, Stargate; **0/5 watched**. | Local HTTP server + actual headless Chrome, visible `#result` and `#profile-result`, watched ID/title set independently compared with displayed titles. |
| Three similar seeds | Star Wars + Empire + Jedi: 3/3 interests covered, Top-5 in both modes has five distinct titles and **0/5 watched**. | Chrome, visible results; score order checked, no errors or warnings in the normal-page console. |
| Three different seeds | Toy Story + Star Wars + Unforgiven: nine profile genres have weight 1/3; Top-5 is Jedi, Empire, Transformers, Kid in King Arthur's Court, Operation Dumbo Drop; **0/5 watched**. Interest coverage is **2/3**: no recommended film has Western, so Unforgiven's interest is lost. | Raw genre intersection of each of five candidates with all three seeds; Chrome now explicitly says `Missing interests: Unforgiven (1992)` while leaving the required pure-cosine ranking intact. |
| Popularity | Catalog median is 27 ratings (both middle counts equal 27). Star Wars + Empire + Jedi: **5/5** recommendations above median (127–429 ratings). Diverse profile: **4/5** above median (22–507 ratings); user 1 profile: **5/5**. | Counts independently computed from `u.data`, confirmed against the visible rating counts and popularity summary for each recommended title in Chrome. Popularity is a measured concentration, not an input to the score; adding a diversity/popularity reranker would change the pure-cosine Top-5. |
| MovieLens history / low ratings | User 1: 272 rated movie **IDs**, 271 distinct titles (both Chasing Amy IDs 246/268 were rated); five distinct recommendations in each mode, **Top-5 ∩ watched IDs/titles = ∅**. A two-star film cannot become a positive seed. | Chrome with user 1, independent `u.data` mapping, Node test with a deliberately injected low-rated selection. One initial audit assertion incorrectly expected 272 distinct **titles**; corrected the assertion after verifying the duplicate IDs rather than changing the filtering logic. |
| User without liked items | User 685 has 20 ratings, none 4–5; neither recommendation mode can be used. | Full `u.data` user audit plus Chrome: both buttons disabled, each panel explains why. |
| Repeated and fast actions | Re-selecting films clears stale comparisons; eight rapid alternating clicks produce the same two Top-5, with zero watched overlap. | Real Chrome buttons clicked repeatedly on the same page; **0** normal-page console errors and **0** uncaught page errors. |
| Damaged dataset | A simulated `u.data` row referencing item 9999 aborts initialization; both panels display the error and recommendation buttons remain disabled. | Browser network override of `u.data`; expected load-failure console logs are distinct from the zero-error normal runs. |

Final commands: `node --test week2/recommender.test.js` → **7/7 pass**; `git diff --check` → pass. Remaining limitation: a strict cosine Top-5 does **not guarantee coverage** of every distinct preference or a balanced distribution of popular and less-popular titles; the UI now measures and reports both rather than claiming these risks are solved. Ratings from all timestamps are used for an illustrative user history, not a time-split offline evaluation.

One further dataset edge case was checked after the table: IDs 246 and 268 both have the title `Chasing Amy (1997)`. Selecting them with ID 50 formerly met the three-ID threshold while supplying only two distinct films to the profile. The profile now rejects that selection with `duplicate IDs of the same title`; the Node suite and a final Chrome run pass. Replacing it with three distinct titles restores both Top-5 lists, 0/5 watched overlap, the 2/3 coverage warning for the diverse example, and 0 normal-page console errors.

## Final HTML and publication preparation

The final HTML check restored the original `id="result-box"` alongside its CSS class and added a data-URI favicon. The latter removed the sole `favicon.ico` HTTP 404 seen in a later independent browser audit. After this change, `node --test week2/recommender.test.js` passed 7/7, `git diff --check` passed, and Chrome displayed five results in both modes with zero console errors. Publication status and the exact revision are determined by the Git branch and commit, not by the earlier local-state notes above.
