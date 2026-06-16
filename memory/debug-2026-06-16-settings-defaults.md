# DEBUG REPORT: Settings And Defaults Locked

- **Symptom:** Users could not adjust processing settings or tell whether "save as defaults" worked.
- **Root cause:** Processing sliders and reset were disabled when no single-image document existed, so batch/default-first workflows were locked. The sliders were also disabled while processing, making large-image auto-processing feel permanently locked. Default saving had no success or failure feedback.
- **Fix:** Processing sliders and reset are editable unless a batch run is active. Saving/restoring defaults now shows explicit success or failure feedback.
- **Evidence:** `npm test` passed with 14 tests. `npm run build` passed.
- **Regression test:** `src/App.test.tsx` verifies processing defaults can be edited before importing an image.
- **Status:** DONE
