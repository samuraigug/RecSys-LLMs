# A01: Random Lunch Generator

This is a revision of the supplied `week1/index.html`, not a new application.
The original source is commit `d8a178a50b1997cd0a5e25604be684b0179418db` in [the course repository](https://github.com/dryjins/RecSys-LLMs).

## Two stages of the fix

The [first fix](https://github.com/samuraigug/RecSys-LLMs/commit/667af2071913435e9cdfb16f7109aad734f0188c) replaced three unsupported Font Awesome 6.4.0 classes. It restored missing glyphs for Ramen, Pasta and Soup. That fix did not establish that all pictures matched their labels: Tacos still showed a spoon, for example.

The final revision uses twelve original inline SVG drawings with one explicit mapping per menu entry. Tacos shows a filled taco, Ramen shows noodles and chopsticks, Steak shows a grilled steak, and BBQ shows a barbecue grill. The two interface icons are local SVGs too; the spinner uses CSS. No remote font, icon library or image is required. Codex authored the SVG markup with the student's approval; these drawings are not copied from a third-party icon pack.

Each click cancels the pending result timer. The newest selection appears after the original 500 ms delay. The twelve menu names, order and selection formula are unchanged. Keyboard activation, focus indication, a status region and reduced-motion styles are included. These additions are not a claim of a full accessibility audit.

## Reproduce a check

1. Open `week1/index.html` in a browser, or serve the repository with `python3 -m http.server 8000` and open `/week1/index.html`.
2. In the browser console set `Math.random = () => 0.45`, then click **Generate Lunch!**. The result should be Ramen with a noodle bowl. Reload to restore normal randomness.
3. For full coverage, set `Math.random = () => (i + 0.25) / 12` for each integer `i` from 0 through 11 and click once. Wait at least 1.1 seconds per item. Check the name, SVG reference, visible drawing and meaning separately.
4. Click five times with roughly 60 ms gaps. After the final click, only the latest choice should leave the loading state. Old pending results must not flash in between.
5. Open the saved HTML file with browser network access disabled. Check all twelve items again. Check a narrow viewport, Tab focus, Enter, Space and the reduced-motion setting.

The model probability is 1/12 for each menu entry under a uniform-random assumption. A deterministic audit is coverage testing, not a sample of random frequencies. Preference weights and negative-feedback learning are not implemented.
