// Run with: node --test week2/recommender.test.js
const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const dir = __dirname;
const rawItems = fs.readFileSync(path.join(dir, 'u.item'));
const rows = new TextDecoder('iso-8859-1').decode(rawItems).trimEnd().split('\n');
const names = [
    'Action', 'Adventure', 'Animation', "Children's", 'Comedy', 'Crime',
    'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical',
    'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'
];
const context = vm.createContext({ TextDecoder, console, window: {} });
vm.runInContext(fs.readFileSync(path.join(dir, 'data.js'), 'utf8'), context);
vm.runInContext(fs.readFileSync(path.join(dir, 'script.js'), 'utf8'), context);

test('all 1682 items have the correct genre coordinates (unknown is not Action)', () => {
    assert.equal(rows.length, 1682);
    const items = vm.runInContext(`parseItemData(${JSON.stringify(rows.join('\n'))}); movies`, context);
    assert.equal(items.length, 1682);
    for (const [index, row] of rows.entries()) {
        const fields = row.split('|');
        assert.equal(fields.length, 24, `row ${index + 1}`);
        const flags = fields.slice(5);
        assert.equal(flags.length, 19);
        assert.ok(flags.every(flag => flag === '0' || flag === '1'));
        const expected = names.filter((_, genreIndex) => flags[genreIndex + 1] === '1');
        assert.equal(items[index].id, Number(fields[0]));
        assert.deepEqual(Array.from(items[index].genres), expected, `movie ${fields[0]}`);
    }
    assert.deepEqual(Array.from(items[0].genres), ['Animation', "Children's", 'Comedy']);
    assert.deepEqual(Array.from(items[202].genres), ['Western']);
    assert.deepEqual(Array.from(items[266].genres), []);
});

test('ISO-8859-1 preserves titles that UTF-8 text() would corrupt', () => {
    assert.match(new TextDecoder().decode(rawItems), /�/);
    const items = vm.runInContext('movies', context);
    assert.equal(items[542].title, 'Misérables, Les (1995)');
    assert.equal(items[1632].title, 'Á köldum klaka (Cold Fever) (1994)');
    assert.ok(items.every(item => !item.title.includes('�')));
});

test('cosine uses both vector norms and handles the empty vector', () => {
    const similarity = vm.runInContext('cosineSimilarity', context);
    assert.ok(Math.abs(similarity([1, 1, 1, 0], [1, 1, 1, 1]) - 3 / Math.sqrt(12)) < 1e-12);
    assert.equal(similarity([0, 0, 0], [1, 0, 1]), 0);
    assert.equal(similarity([0, 0, 0], [0, 0, 0]), 0);
    const actual = vm.runInContext('cosineSimilarity(movieVector(movies[0]), movieVector(movies[94]))', context);
    assert.ok(Math.abs(actual - 3 / Math.sqrt(12)) < 1e-12);
});

test('a mixed three-film profile averages coordinates rather than merging genres', () => {
    const profile = vm.runInContext('averageProfile([movies[0], movies[49], movies[202]])', context);
    const coordinate = genre => profile[names.indexOf(genre)];
    assert.equal(coordinate('Animation'), 1 / 3); // Toy Story only
    assert.equal(coordinate('Action'), 1 / 3);    // Star Wars only
    assert.equal(coordinate('Western'), 1 / 3);   // Unforgiven only
    assert.equal(coordinate('Drama'), 0);
    const overlapping = vm.runInContext('averageProfile([movies[0], movies[94], movies[202]])', context);
    assert.equal(overlapping[names.indexOf('Animation')], 2 / 3);
    assert.equal(overlapping[names.indexOf('Musical')], 1 / 3);
    const score = vm.runInContext('cosineSimilarity(averageProfile([movies[0], movies[94], movies[202]]), movieVector(movies[421]))', context);
    assert.ok(Math.abs(score - 6 / Math.sqrt(42)) < 1e-12);
    // Two unknown-only items would merely rescale a one-film vector; cosine cancels that factor.
    const deceptiveScore = vm.runInContext('cosineSimilarity(averageProfile([movies[266], movies[1372], movies[0]]), movieVector(movies[94]))', context);
    const singleScore = vm.runInContext('cosineSimilarity(movieVector(movies[0]), movieVector(movies[94]))', context);
    assert.ok(Math.abs(deceptiveScore - singleScore) < 1e-12);
});

test('all 100000 ratings have valid user, movie, score, timestamp and unique pair', () => {
    const lines = fs.readFileSync(path.join(dir, 'u.data'), 'utf8').trimEnd().split('\n');
    assert.equal(lines.length, 100000);
    const parsed = vm.runInContext(`parseRatingData(${JSON.stringify(lines.join('\n'))}); ratings`, context);
    assert.equal(parsed.length, lines.length);
    const itemIds = new Set(vm.runInContext('movies', context).map(movie => movie.id));
    const users = new Set();
    const pairs = new Set();
    lines.forEach((line, index) => {
        const fields = line.split('\t').map(Number);
        assert.equal(fields.length, 4, `rating row ${index + 1}`);
        const [userId, itemId, rating, timestamp] = fields;
        assert.ok(Number.isInteger(userId) && userId > 0);
        assert.ok(itemIds.has(itemId));
        assert.ok(Number.isInteger(rating) && rating >= 1 && rating <= 5);
        assert.ok(Number.isInteger(timestamp) && timestamp >= 0);
        assert.equal(pairs.has(`${userId}:${itemId}`), false);
        pairs.add(`${userId}:${itemId}`);
        users.add(userId);
        assert.equal(parsed[index].userId, userId);
        assert.equal(parsed[index].itemId, itemId);
        assert.equal(parsed[index].rating, rating);
        assert.equal(parsed[index].timestamp, timestamp);
    });
    assert.equal(users.size, 943);
    assert.equal(pairs.size, 100000);
});

test('malformed records cannot silently drop watched history', () => {
    for (const row of ['1\t1\t0\t1', '1\t9999\t4\t1', '1\t1\t4\t1\n1\t1\t5\t2', '1\t1\t4', '1\t1\t4\t']) {
        assert.throws(() => vm.runInContext(`parseRatingData(${JSON.stringify(row)})`, context), /Invalid u.data record/);
    }
    assert.equal(vm.runInContext('ratings.length', context), 100000);
    const badItem = rows[0].replace(/\|0$/, '|2');
    assert.throws(() => vm.runInContext(`parseItemData(${JSON.stringify(badItem)})`, context), /Invalid u.item record/);
    assert.equal(vm.runInContext('movies.length', context), 1682);
});

test('all rated films are excluded, even when a 2-star film matches a liked seed', () => {
    vm.runInContext(`movies.forEach(movie => movieById.set(movie.id, movie));
        ratings.forEach(rating => {
            if (!ratingsByUser.has(rating.userId)) ratingsByUser.set(rating.userId, []);
            ratingsByUser.get(rating.userId).push(rating);
        });`, context);
    const elements = {
        'user-select': { value: '1' },
        'movie-select': { value: '20' }, // rated 4 by user 1
        'profile-select': { selectedOptions: [{ value: '50' }, { value: '172' }, { value: '181' }] },
        'profile-result': { textContent: '', className: '' },
        'result': { textContent: '', className: '' }
    };
    context.document = { getElementById: id => elements[id] };
    const history = vm.runInContext('watchedMovieIds()', context);
    assert.ok(history.has(36)); // Mad Love was rated 2 by user 1
    assert.ok(history.has(20));
    assert.equal(vm.runInContext('ratingsByUser.get(1).find(r => r.itemId === 36).rating', context), 2);
    assert.equal(vm.runInContext('ratingsByUser.get(1).find(r => r.itemId === 20).rating', context), 4);
    assert.equal(vm.runInContext('ratingsByUser.get(685).filter(r => r.rating >= 4).length', context), 0);
    elements['movie-select'].value = '36';
    vm.runInContext('getRecommendations()', context);
    assert.match(elements.result.textContent, /rating of 4 or 5/);
    elements['movie-select'].value = '20';

    const single = vm.runInContext('topFive(movieVector(movieById.get(20)), watchedMovieIds())', context);
    const profile = vm.runInContext('topFive(averageProfile([movieById.get(50), movieById.get(172), movieById.get(181)]), watchedMovieIds())', context);
    const items = vm.runInContext('movieById', context);
    const watchedTitles = new Set([...history].map(id => items.get(id)?.title));
    for (const list of [single, profile]) {
        assert.equal(list.length, 5);
        assert.ok(list.every(movie => !history.has(movie.id)));
        assert.ok(list.every(movie => !watchedTitles.has(movie.title)));
    }

    const diverse = vm.runInContext(`topFive(
        averageProfile([movieById.get(1), movieById.get(50), movieById.get(203)]),
        new Set([1, 50, 203]))`, context);
    const uncovered = vm.runInContext('uncoveredProfileFilms', context)([items.get(1), items.get(50), items.get(203)], diverse);
    assert.deepEqual(uncovered.map(movie => movie.id), [203]);

    // Even a manually injected low-rated selection must not enter the positive profile.
    elements['profile-select'].selectedOptions = [{ value: '20' }, { value: '36' }, { value: '50' }];
    vm.runInContext('getProfileRecommendations()', context);
    assert.match(elements['profile-result'].textContent, /rating of 4 or 5/);
    elements['profile-select'].selectedOptions = [{ value: '20' }, { value: '50' }];
    vm.runInContext('getProfileRecommendations()', context);
    assert.match(elements['profile-result'].textContent, /at least three/);
    elements['profile-select'].selectedOptions = [{ value: '267' }, { value: '1373' }, { value: '50' }];
    vm.runInContext('getProfileRecommendations()', context);
    assert.match(elements['profile-result'].textContent, /known genres/);
    elements['profile-select'].selectedOptions = [{ value: '246' }, { value: '268' }, { value: '50' }];
    vm.runInContext('getProfileRecommendations()', context);
    assert.match(elements['profile-result'].textContent, /duplicate IDs/);
});
