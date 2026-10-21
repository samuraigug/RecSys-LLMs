// ---------------------------------------------------------------------------
// week6/data.js — MovieLens 100k data layer (classic <script>, no modules)
//
// Loads u.item (movies) and u.data (ratings), builds a sparse-friendly
// rating matrix, and splits every user's ratings chronologically 80/10/10.
//
// Genre alignment note (fixes the Week-2 off-by-one):
//   A u.item row has 24 '|'-separated fields:
//     0 id | 1 title | 2 release | 3 video | 4 url | 5 unknown |
//     6 Action | 7 Adventure | ... | 23 Western
//   Field index 5 is the legacy 19th genre flag "unknown", NOT the first real
//   genre. The 18 real flags therefore live at indices 6..23, so we read them
//   with fields.slice(6, 24) and pair them with the 18-name `genreNames`
//   list below. (The earlier fields.slice(5, 24) + 18 names pairing was off
//   by one.)
// ---------------------------------------------------------------------------

// Global variables for storing movie and rating data
let movies = [];          // [{ id, title, genres: string[], unknownGenre: boolean }]
let ratings = [];         // [{ userId, movieId, rating, timestamp }]

// Collaborative filtering structures (populated by buildRatingMatrix)
let numUsers = 0;         // highest user id found in u.data
let numMovies = 0;        // number of parsed movies
let ratingMatrix = null;  // (numUsers + 1) x (maxMovieId + 1); 0 = unobserved

// Genre names for the 18 real flags (u.item field indices 6..23)
const genreNames = [
    "Action", "Adventure", "Animation", "Children's", "Comedy",
    "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir",
    "Horror", "Musical", "Mystery", "Romance", "Sci-Fi",
    "Thriller", "War", "Western"
];

// Users skipped while splitting (e.g. too few ratings), collected by
// splitByTimestamp() for reporting.
let SPLIT_EXCLUSIONS = [];
let DATA_SOURCE = null;   // Hashes of original fetched bytes, not decoded text.

/** Exact ISO-8859-1 decoding; unlike WHATWG latin1, preserves bytes 0x80..0x9f. */
function decodeItemBytes(buffer) {
    const bytes = new Uint8Array(buffer);
    const chunks = [];
    for (let i = 0; i < bytes.length; i += 8192) {
        chunks.push(String.fromCharCode(...bytes.subarray(i, i + 8192)));
    }
    return chunks.join('');
}

/** SHA-256 of bytes, available on localhost/HTTPS without a dependency. */
async function dataSHA256(bytes) {
    const digest = await crypto.subtle.digest('SHA-256', bytes);
    return Array.from(new Uint8Array(digest), n => n.toString(16).padStart(2, '0')).join('');
}

// Primary function to load data from files
async function loadData() {
    try {
        const [moviesResponse, ratingsResponse] = await Promise.all([fetch('u.item'), fetch('u.data')]);
        if (!moviesResponse.ok) {
            throw new Error(`Failed to load movie data: ${moviesResponse.status}`);
        }
        if (!ratingsResponse.ok) {
            throw new Error(`Failed to load rating data: ${ratingsResponse.status}`);
        }
        const [itemBytes, ratingBytes] = await Promise.all([moviesResponse.arrayBuffer(), ratingsResponse.arrayBuffer()]);
        // Parse/validate in isolation: a failed request or parse preserves the old dataset.
        const nextMovies = parseItemData(decodeItemBytes(itemBytes), false);
        const nextRatings = parseRatingData(new TextDecoder('utf-8', { fatal: true }).decode(ratingBytes), false);
        const catalogIds = new Set(nextMovies.map(movie => movie.id));
        if (nextRatings.some(row => !catalogIds.has(row.movieId))) {
            throw new Error('A rating references a movie absent from u.item.');
        }
        const nextUsers = nextRatings.reduce((max, r) => Math.max(max, r.userId), 0);
        const nextMatrix = buildRatingMatrix(nextMovies, nextRatings, nextUsers, false);
        const [itemHash, ratingHash] = await Promise.all([dataSHA256(itemBytes), dataSHA256(ratingBytes)]);
        movies = nextMovies;
        ratings = nextRatings;
        numUsers = nextUsers;
        numMovies = movies.length;
        ratingMatrix = nextMatrix;
        DATA_SOURCE = {
            'u.item': { sha256: itemHash, bytes: itemBytes.byteLength, encoding: 'ISO-8859-1' },
            'u.data': { sha256: ratingHash, bytes: ratingBytes.byteLength, encoding: 'UTF-8 (ASCII)' }
        };
        document.dispatchEvent(new Event('hw6-data-loaded'));
        return DATA_SOURCE;
    } catch (error) {
        console.error('Error loading data:', error);
        const errorTarget = document.getElementById('result');
        if (errorTarget) {
            errorTarget.textContent =
                `Error: ${error.message}. Please make sure u.item and u.data are in the correct location.`;
            errorTarget.className = 'error';
        }
        throw error; // Re-throw so the page script can handle the error
    }
}

// Parse movie data from u.item format: "id|title|release|video|url|g0|...|g18"
// genreNames has 18 entries, matched against the real flags at indices 6..23.
function parseItemData(text, commit = true) {
    const lines = text.split('\n');
    const parsed = [], ids = new Set();
    for (const line of lines) {
        if (line.trim() === '') continue;

        const fields = line.split('|');
        if (fields.length !== 24) throw new Error('u.item row must have 24 fields.');
        const id = Number(fields[0]);
        if (!Number.isSafeInteger(id) || id <= 0 || ids.has(id)) throw new Error('Invalid or duplicate movie ID: ' + fields[0]);
        if (fields.slice(5, 24).some(flag => !/^[01]$/.test(flag.trim()))) throw new Error('Invalid genre flag for movie ' + id);
        ids.add(id);
        const title = fields[1];

        // Read the 18 real genre flags (index 5 is the 'unknown' placeholder)
        const genreValues = fields.slice(6, 24).map(value => parseInt(value));
        const genres = genreNames.filter((_, index) => genreValues[index] === 1);

        parsed.push({ id, title, genres, unknownGenre: Number(fields[5]) === 1 });
    }
    if (commit) movies = parsed;
    return parsed;
}

// Parse rating data from u.data format: "userId\tmovieId\trating\ttimestamp"
function parseRatingData(text, commit = true) {
    const lines = text.split('\n');
    const parsed = [];
    for (const line of lines) {
        if (line.trim() === '') continue;

        const fields = line.split('\t');
        if (fields.length !== 4) throw new Error('u.data row must have 4 fields.');
        const [userId, movieId, rating, timestamp] = fields.map(Number);
        if (![userId, movieId].every(id => Number.isSafeInteger(id) && id > 0) ||
            !Number.isInteger(rating) || rating < 1 || rating > 5 ||
            !Number.isSafeInteger(timestamp) || timestamp < 0) throw new Error('Invalid rating record: ' + line);
        parsed.push({ userId, movieId, rating, timestamp });
    }
    if (commit) ratings = parsed;
    return parsed;
}

// Build the sparse-friendly rating matrix.
// Shape: (numUsers + 1) x (maxMovieId + 1), indexed by raw ids, so that
//   ratingMatrix[userId][movieId] === rating
// and an unobserved pair is 0. MovieLens ratings are 1-5, so 0 is unambiguous.
function buildRatingMatrix(movieList = movies, ratingList = ratings, userCount = numUsers, commit = true) {
    const maxMovieId = movieList.reduce((max, m) => Math.max(max, m.id), 0);

    const matrix = [];
    for (let u = 0; u <= userCount; u++) {
        matrix.push(new Array(maxMovieId + 1).fill(0));
    }

    for (const r of ratingList) {
        if (r.userId <= userCount && r.movieId <= maxMovieId) {
            matrix[r.userId][r.movieId] = r.rating;
        }
    }
    if (commit) ratingMatrix = matrix;
    return matrix;
}

// Split EACH user's ratings by ascending timestamp into 80/10/10.
// Returns { train, val, test }, each an array of the same
// { userId, movieId, rating, timestamp } objects.
//
// Rules:
//   - counts are deterministic: nTrain = floor(0.8 * n), nVal = floor(0.1 * n),
//     nTest = n - nTrain - nVal (the remainder)
//   - users with fewer than 5 ratings are skipped and recorded in
//     SPLIT_EXCLUSIONS
//   - timestamp ties are broken by movieId so the split is deterministic
function splitByTimestamp() {
    const MIN_RATINGS = 5;

    const byUser = new Map();
    for (const r of ratings) {
        if (!byUser.has(r.userId)) byUser.set(r.userId, []);
        byUser.get(r.userId).push(r);
    }

    const train = [];
    const val = [];
    const test = [];

    SPLIT_EXCLUSIONS = [];

    for (const [userId, userRatings] of byUser) {
        const n = userRatings.length;
        if (n < MIN_RATINGS) {
            SPLIT_EXCLUSIONS.push({
                userId,
                count: n,
                reason: `fewer than ${MIN_RATINGS} ratings`
            });
            continue;
        }

        const sorted = userRatings.slice().sort((a, b) =>
            a.timestamp - b.timestamp || a.movieId - b.movieId
        );

        const nTrain = Math.floor(0.8 * n);
        const nVal = Math.floor(0.1 * n);

        for (let i = 0; i < n; i++) {
            if (i < nTrain) train.push(sorted[i]);
            else if (i < nTrain + nVal) val.push(sorted[i]);
            else test.push(sorted[i]);
        }
    }

    return { train, val, test };
}
