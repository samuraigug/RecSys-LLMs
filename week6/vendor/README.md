# Browser dependencies

- `d3.min.js`: original instructor vendor, d3 **7.9.0**, unchanged.
- `tf-4.22.0.min.js`: TensorFlow.js **4.22.0**, byte-identical to the genuine
  jsDelivr response captured during stage1. Copyright/license header retained;
  Apache2.0 terms are supplied in `TFJS-LICENSE-APACHE-2.0.txt`.

Original URL:
`https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.22.0/dist/tf.min.js`

Stage4 pinned the same library locally after repeated observed CDN socket errors.
This is not a version upgrade, alternate implementation, or mocked calculation.
Hashes/provenance are in `evidence/a06-stage4/tf-vendor-provenance.json` at repository
root and the stage manifests. The runtime uses only relative local dependency and
model URLs; no dependency install or external CDN availability is required.
