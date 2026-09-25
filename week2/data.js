// Global variables for storing movie and rating data
let movies = [];
let ratings = [];

// Genre names as defined in the u.item file
const genreNames = [
    "Action", "Adventure", "Animation", "Children's", "Comedy",
    "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir",
    "Horror", "Musical", "Mystery", "Romance", "Sci-Fi",
    "Thriller", "War", "Western"
];

// Primary function to load data from files
async function loadData() {
    try {
        // Load and parse movie data
        const moviesResponse = await fetch('u.item');
        if (!moviesResponse.ok) {
            throw new Error(`Failed to load movie data: ${moviesResponse.status}`);
        }
        // MovieLens 100K u.item is ISO-8859-1, not UTF-8 (fetch.text() corrupts accented titles).
        const moviesText = new TextDecoder('iso-8859-1').decode(await moviesResponse.arrayBuffer());
        parseItemData(moviesText);

        // Load and parse rating data
        const ratingsResponse = await fetch('u.data');
        if (!ratingsResponse.ok) {
            throw new Error(`Failed to load rating data: ${ratingsResponse.status}`);
        }
        const ratingsText = await ratingsResponse.text();
        parseRatingData(ratingsText);
    } catch (error) {
        console.error('Error loading data:', error);
        const resultElement = document.getElementById('result');
        if (resultElement) {
            resultElement.textContent = `Error: ${error.message}. Please make sure u.item and u.data files are in the correct location.`;
            resultElement.className = 'error';
        }
        throw error; // Re-throw to allow script.js to handle the error
    }
}

// Parse movie data from u.item format
function parseItemData(text) {
    const lines = text.split('\n');
    const parsedMovies = [];
    const seenIds = new Set();
    
    for (const [index, line] of lines.entries()) {
        if (line.trim() === '') continue;
        
        const fields = line.split('|');
        const id = Number(fields[0]);
        const title = fields[1];
        const flags = fields.slice(5);
        if (fields.length !== 24 || !Number.isSafeInteger(id) || id < 1 ||
            !title?.trim() || seenIds.has(id) || flags.some(flag => flag !== '0' && flag !== '1')) {
            throw new Error(`Invalid u.item record at line ${index + 1}`);
        }
        seenIds.add(id);

        // Field 5 is "unknown"; the 18 named genres start at field 6.
        const genreValues = fields.slice(6, 24).map(Number);
        const genres = genreNames.filter((_, index) => genreValues[index] === 1);
        
        parsedMovies.push({ id, title, genres });
    }
    movies = parsedMovies;
}

// Parse rating data from u.data format
function parseRatingData(text) {
    const lines = text.split('\n');
    const parsedRatings = [];
    const itemIds = new Set(movies.map(movie => movie.id));
    const seenPairs = new Set();
    
    for (const [index, line] of lines.entries()) {
        if (line.trim() === '') continue;
        
        const fields = line.split('\t');
        const [userId, itemId, rating, timestamp] = fields.map(Number);
        const pair = `${userId}:${itemId}`;
        if (fields.length !== 4 || fields.some(field => field.trim() === '') ||
            !Number.isSafeInteger(userId) || userId < 1 ||
            !itemIds.has(itemId) || !Number.isInteger(rating) || rating < 1 || rating > 5 ||
            !Number.isSafeInteger(timestamp) || timestamp < 0 || seenPairs.has(pair)) {
            throw new Error(`Invalid u.data record at line ${index + 1}`);
        }
        seenPairs.add(pair);
        
        parsedRatings.push({ userId, itemId, rating, timestamp });
    }
    ratings = parsedRatings;
}
