// Keep all rated films (including low ratings) separate from positively rated films.
const ratingsByUser = new Map();
const movieById = new Map();
const ratingCountByMovie = new Map();
let catalogMedianRatings = 0;

window.onload = async function() {
    try {
        const result = document.getElementById('result');
        result.textContent = 'Loading movie data...';
        result.className = 'loading';
        await loadData();

        movies.forEach(movie => movieById.set(movie.id, movie));
        ratings.forEach(rating => {
            if (!ratingsByUser.has(rating.userId)) ratingsByUser.set(rating.userId, []);
            ratingsByUser.get(rating.userId).push(rating);
            ratingCountByMovie.set(rating.itemId, (ratingCountByMovie.get(rating.itemId) || 0) + 1);
        });
        const counts = movies.map(movie => ratingCountByMovie.get(movie.id) || 0).sort((a, b) => a - b);
        catalogMedianRatings = counts[Math.floor(counts.length / 2)];
        populateUserDropdown();
        populateMoviesDropdown();
        for (const id of ['user-select', 'movie-select', 'profile-select']) {
            document.getElementById(id).disabled = false;
        }
        clearRecommendations();
    } catch (error) {
        console.error('Initialization error:', error);
        // The loading error is already displayed by data.js in the single-film panel.
        const profile = document.getElementById('profile-result');
        profile.textContent = `Data loading failed: ${error.message}`;
        profile.className = 'error';
    }
};

function populateUserDropdown() {
    const select = document.getElementById('user-select');
    [...ratingsByUser.keys()].sort((a, b) => a - b).forEach(userId => {
        const option = document.createElement('option');
        option.value = userId;
        option.textContent = `User ${userId}`;
        select.appendChild(option);
    });
}

function getUserRatings() {
    const userId = document.getElementById('user-select').value;
    return userId ? ratingsByUser.get(Number(userId)) || [] : null;
}

function populateMoviesDropdown() {
    const singleSelect = document.getElementById('movie-select');
    const profileSelect = document.getElementById('profile-select');
    while (singleSelect.options.length > 1) singleSelect.remove(1);
    profileSelect.replaceChildren();
    singleSelect.selectedIndex = 0;

    const userRatings = getUserRatings();
    const likedIds = userRatings && new Set(userRatings.filter(r => r.rating >= 4).map(r => r.itemId));
    const available = [...movies]
        .filter(movie => !likedIds || likedIds.has(movie.id))
        .sort((a, b) => a.title.localeCompare(b.title) || a.id - b.id);

    available.forEach(movie => {
        for (const select of [singleSelect, ...(movie.genres.length ? [profileSelect] : [])]) {
            const option = document.createElement('option');
            option.value = movie.id;
            option.textContent = movie.title;
            select.appendChild(option);
        }
    });
}

function clearRecommendations() {
    const single = document.getElementById('result');
    const profile = document.getElementById('profile-result');
    single.textContent = 'Select one liked film.';
    single.className = '';
    profile.textContent = 'Select at least three liked films.';
    profile.className = '';
    const userId = document.getElementById('user-select').value;
    const userRatings = getUserRatings();
    const positiveCount = userRatings && userRatings.filter(r => r.rating >= 4).length;
    const profileCount = userRatings && userRatings.filter(r => r.rating >= 4 && movieById.get(r.itemId)?.genres.length).length;
    document.getElementById('recommend-btn').disabled = !!userRatings && positiveCount === 0;
    document.getElementById('profile-btn').disabled = !!userRatings && profileCount < 3;
    if (userRatings && positiveCount === 0) {
        single.textContent = `User ${userId} has no films rated 4–5; a liked-film recommendation is unavailable.`;
    }
    if (userRatings && profileCount < 3) {
        profile.textContent = `User ${userId} has ${profileCount} film(s) rated 4–5 with known genres; at least three are required for a profile.`;
    }
}

function changeUser() {
    populateMoviesDropdown();
    clearRecommendations();
}

// Binary genre vectors share the same coordinate order as genreNames.
function movieVector(movie) {
    const genres = new Set(movie.genres);
    return genreNames.map(genre => genres.has(genre) ? 1 : 0);
}

// Mean of the liked films' vectors: a genre present in 2 of 3 films has weight 2/3.
function averageProfile(likedMovies) {
    const vectors = likedMovies.map(movieVector);
    return genreNames.map((_, index) => vectors.reduce((sum, vector) => sum + vector[index], 0) / vectors.length);
}

function cosineSimilarity(a, b) {
    const dot = a.reduce((sum, value, index) => sum + value * b[index], 0);
    const normA = Math.sqrt(a.reduce((sum, value) => sum + value * value, 0));
    const normB = Math.sqrt(b.reduce((sum, value) => sum + value * value, 0));
    return normA && normB ? dot / (normA * normB) : 0;
}

// Include all rated items for a selected MovieLens user, even 1-star and 2-star items.
// With manual selection, the single-film seed and all profile seeds are the known watched films.
function watchedMovieIds() {
    const watched = new Set((getUserRatings() || []).map(rating => rating.itemId));
    const singleId = Number(document.getElementById('movie-select').value);
    if (singleId) watched.add(singleId);
    for (const option of document.getElementById('profile-select').selectedOptions) {
        watched.add(Number(option.value));
    }
    return watched;
}

function topFive(vector, watched) {
    // Several MovieLens IDs have the same title; exclude their other copies too.
    const watchedTitles = new Set([...watched].map(id => movieById.get(id)?.title));
    const scored = movies
        .filter(movie => !watched.has(movie.id) && !watchedTitles.has(movie.title))
        .map(movie => ({ ...movie, score: cosineSimilarity(vector, movieVector(movie)) }))
        .filter(movie => movie.score > 0)
        .sort((a, b) => b.score - a.score || a.id - b.id);

    const seenTitles = new Set();
    const result = [];
    for (const movie of scored) {
        if (seenTitles.has(movie.title)) continue;
        seenTitles.add(movie.title);
        result.push(movie);
        if (result.length === 5) break;
    }
    return result;
}

// Diagnostic only: changing these results would no longer be a pure cosine Top-5.
function uncoveredProfileFilms(likedMovies, recommendations) {
    return likedMovies.filter(seed => !recommendations.some(movie =>
        seed.genres.some(genre => movie.genres.includes(genre))));
}

function displayRecommendations(target, likedMovies, vector, profileMode) {
    const watched = watchedMovieIds();
    const recommendations = topFive(vector, watched);
    const userId = document.getElementById('user-select').value;
    const source = likedMovies.map(movie => `• ${movie.title}`).join('\n');
    const context = userId ? `MovieLens user ${userId}` : 'Manual selection';
    let explanation = `${profileMode ? 'Profile (mean of genre vectors)' : 'One-film recommendations'}\nBased on:\n${source}\n${context}: ${watched.size} watched movie IDs excluded.`;

    if (profileMode) {
        const weights = genreNames.map((genre, index) => ({ genre, weight: vector[index] }))
            .filter(entry => entry.weight > 0)
            .map(entry => `${entry.genre} ${entry.weight.toFixed(2)}`);
        explanation += `\nProfile genres: ${weights.join(', ') || 'none'}.`;
        if (!genreNames.some(genre => likedMovies.every(movie => movie.genres.includes(genre)))) {
            explanation += '\nNote: no genre is shared by all selected films; this mixed profile may give broad recommendations.';
        }
        const uncovered = uncoveredProfileFilms(likedMovies, recommendations);
        explanation += `\nInterest coverage in Top-5: ${likedMovies.length - uncovered.length}/${likedMovies.length} selected films share a genre with at least one recommendation.`;
        if (uncovered.length) {
            explanation += `\nMissing interests: ${uncovered.map(movie => movie.title).join('; ')}. The ranking remains pure cosine.`;
        }
    }

    if (!recommendations.length) {
        target.textContent = `${explanation}\nNo recommendations with positive similarity.`;
        target.className = 'error';
        return;
    }
    const aboveMedian = recommendations.filter(movie => (ratingCountByMovie.get(movie.id) || 0) > catalogMedianRatings).length;
    explanation += `\nPopularity audit: ${aboveMedian}/${recommendations.length} above the catalog median (${catalogMedianRatings} ratings); popularity does not affect the cosine ranking.`;
    const lines = recommendations.map((movie, index) =>
        `${index + 1}. ${movie.title} — cosine ${movie.score.toFixed(3)}; ${ratingCountByMovie.get(movie.id) || 0} ratings`);
    target.textContent = `${explanation}\nTop-${recommendations.length}:\n${lines.join('\n')}`;
    target.className = 'success';
}

function getRecommendations() {
    const result = document.getElementById('result');
    const id = Number(document.getElementById('movie-select').value);
    const movie = movieById.get(id);
    if (!movie) {
        result.textContent = 'Please select a movie first.';
        result.className = 'error';
        return;
    }
    const userRatings = getUserRatings();
    if (userRatings && !userRatings.some(r => r.itemId === id && r.rating >= 4)) {
        result.textContent = 'The selected film must have a rating of 4 or 5 for this user.';
        result.className = 'error';
        return;
    }
    if (!movie.genres.length) {
        result.textContent = `No recommendations found for "${movie.title}": no known genres.`;
        result.className = 'error';
        return;
    }
    displayRecommendations(result, [movie], movieVector(movie), false);
}

function getProfileRecommendations() {
    const result = document.getElementById('profile-result');
    const selectedIds = [...document.getElementById('profile-select').selectedOptions].map(option => Number(option.value));
    if (new Set(selectedIds).size < 3) {
        result.textContent = 'Please select at least three different liked films.';
        result.className = 'error';
        return;
    }
    const likedMovies = selectedIds.map(id => movieById.get(id));
    if (likedMovies.some(movie => !movie || !movie.genres.length)) {
        result.textContent = 'Profile films must have known genres; unknown-only films cannot describe a preference.';
        result.className = 'error';
        return;
    }
    if (new Set(likedMovies.map(movie => movie.title)).size < 3) {
        result.textContent = 'Please select at least three different films, not duplicate IDs of the same title.';
        result.className = 'error';
        return;
    }
    const userRatings = getUserRatings();
    const positiveIds = userRatings && new Set(userRatings.filter(r => r.rating >= 4).map(r => r.itemId));
    if (likedMovies.some((movie, index) => positiveIds && !positiveIds.has(selectedIds[index]))) {
        result.textContent = 'Profile films must have a rating of 4 or 5 for the selected user.';
        result.className = 'error';
        return;
    }
    displayRecommendations(result, likedMovies, averageProfile(likedMovies), true);
}
