# Previous selected HW5 matrix factorization

These assets are Gurgen Avetisyan's original HW5 checkpoint, not the instructor's
HW6 MF reference. Original run and selection-lock bytes are retained unchanged.
The selected epoch is **24** from a 40-epoch run, F=16, lr=.01, reg=.0001,
batch64; selection used validation RMSE. No model is retrained in HW6.

P and Q are base64-encoded little-endian float32 with raw-ID row addressing.
Shapes include the original unused ID0 row; HW6 graph indices are separate.
Scoring remains `clip(mu + dot(P[u],Q[i]),1,5)`, ties by ascending movie ID.
The historical full-matrix-L2 objective is not silently changed.

`hw5-manifest.json` attributes the source and pins files, decoded factors,
dataset hashes and model fingerprint. Runtime loads only relative URLs here;
it has no dependency on the original author's filesystem paths.
