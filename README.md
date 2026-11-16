# SETI Narrowband Signal Detection: Classical Search vs. Machine Learning

An undergraduate research project exploring how well a standard SETI search
algorithm detects faint drifting radio signals, and whether a small neural
network can do better.

**Status:** in progress. All results so far are on simulated data.

## Key findings so far
- On simple **straight** drifting signals, a classical straight-line drift
  search (the core idea behind `turbo_seti`) beats a small CNN
  (ROC AUC 0.951 vs. 0.923), even with 10,000 training frames.
- On signals whose **path bends**, the CNN beats that same search
  (AUC 0.945 vs. 0.914 for curved, wobbling signals), consistently across
  3 seeds.
- **Brightness changes alone** do not help the CNN: with a straight path,
  the two methods are about even. The bending path is what breaks the
  straight-line search.
- Next question: does the CNN still win against a classical search that
  also tries curved paths, and on real telescope noise?

## Background
SETI searches look for narrowband radio signals that drift in frequency
because of the Doppler effect. Breakthrough Listen's `turbo_seti` is a
standard tool for finding them. Very faint signals are hard to separate from
random noise: set the detection threshold too high and real signals are
missed, set it too low and noise triggers false alarms.

## What's done so far
1. **Real data:** loaded Breakthrough Listen's Voyager 1 observation with
   `blimpy` and plotted its carrier and sidebands.
2. **Classical search:** ran `turbo_seti` on that file and recovered the
   carrier (drift of about -0.38 Hz/s, SNR of about 246).
3. **Controlled test bench:** injected fake signals with known drift and SNR
   into noise using `setigen`, then checked what `turbo_seti` recovers.
   - With the search threshold at SNR 10, signals injected at SNR 9 and above
     were found, and weaker ones were not (single trial per setting).
   - Lowering the threshold to 5 produced a large number of false hits from
     noise.
4. **ML dataset:** generated 3,000 labeled spectrograms (half with a hidden
   drifting signal, half noise only) with random SNR from 3 to 15.
5. **ML model:** trained a small convolutional network (3 conv layers,
   15 epochs) to classify them. On a held-out test set of 600 spectrograms
   (20% of the data, never used in training) it reached about 85% accuracy.

   The network outputs a probability that a signal is present; the
   **cutoff** is the probability above which we call it a detection.
   Raising the cutoff trades sensitivity for fewer false alarms.
   **False alarms** is the fraction of the 300 noise-only test examples
   flagged as signals; the **SNR** columns are the fraction of injected
   signals detected in each SNR range (62, 83 and 70 signals respectively).
   At the default cutoff of 0.5, all 85 signals with SNR 12-15 were detected.
   Single training run, so these numbers are preliminary.

   | Cutoff | False alarms | SNR 3-6 | SNR 6-9 | SNR 9-12 |
   |-------:|-------------:|--------:|--------:|---------:|
   | 0.3    | 18.3%        | 50%     | 88%     | 100%     |
   | 0.5    | 8.7%         | 29%     | 76%     | 100%     |
   | 0.7    | 3.3%         | 16%     | 63%     | 97%      |
   | 0.9    | 0.3%         | 6%      | 53%     | 89%      |

6. **Fair classical baseline:** comparing the CNN to `turbo_seti` directly
   was not like for like (different file sizes and thresholds). So I wrote a
   small drift search that runs on the same 32×256 test frames the CNN sees.
   It adds up the power along every straight drift line through the frame
   (covering the full injected drift range) and keeps the strongest one,
   which is the core idea behind `turbo_seti`.

   Both methods give one score per frame, so sweeping a cutoff over each
   gives an ROC curve (detection rate vs. false alarm rate). For a per-SNR
   comparison, each method's cutoff is set so that both flag exactly 5% of
   the 300 noise-only test frames:

   | Method       | ROC AUC | SNR 3-6 | SNR 6-9 | SNR 9-12 | SNR 12-15 |
   |--------------|--------:|--------:|--------:|---------:|----------:|
   | CNN          | 0.924   | 21%     | 77%     | 96%      | 100%      |
   | Drift search | 0.953   | 45%     | 96%     | 100%     | 100%      |

   **The simple drift search beats the CNN**, and the gap is largest for
   faint signals: at SNR 3-6 it finds about twice as many. A separate test
   run gave the same picture (AUC 0.950 vs. 0.927). This is not surprising:
   the synthetic signals are perfectly straight lines in plain noise, which
   is exactly the case summing along straight lines is built for. For the
   CNN to be worth using, it probably needs data where that assumption
   breaks down.

   Note: these early numbers came from unseeded runs, so they varied from
   run to run, and the cutoff table above and this comparison come from
   different runs. The notebook is now seeded, and item 7 below repeats
   the comparison more carefully.

7. **Larger, repeated comparison on harder signals:** redid the comparison
   with 10,000 training frames, longer training (best epoch kept on a
   validation set), 3 seeds, and 5% cutoffs set on a separate set of
   noise-only frames, so the test frames are never used to tune anything.
   Then repeated it for signals the straight-line search was not built for:
   paths that bend (**curved**), brightness that swings ±60% around its
   average (**wobble**), and both. Each test set has 2,000 frames.
   Numbers are the mean ± spread over the 3 seeds; detection rates are at
   a matched 5% false alarm rate.

   | Signals       | Method       | ROC AUC       | SNR 3-6     | SNR 6-9     |
   |---------------|--------------|---------------|-------------|-------------|
   | Straight      | CNN          | 0.923 ± 0.008 | 31% ± 2     | 84% ± 2     |
   |               | Drift search | 0.951 ± 0.004 | 49% ± 3     | 100% ± 1    |
   | Curved+wobble | CNN          | 0.945 ± 0.004 | 42% ± 1     | 93% ± 2     |
   |               | Drift search | 0.914 ± 0.009 | 26% ± 4     | 80% ± 2     |
   | Curved only   | CNN          | 0.925 ± 0.006 | 29% ± 4     | 88% ± 2     |
   |               | Drift search | 0.906 ± 0.010 | 25% ± 3     | 73% ± 4     |
   | Wobble only   | CNN          | 0.950 ± 0.004 | 48% ± 3     | 97% ± 2     |
   |               | Drift search | 0.946 ± 0.003 | 45% ± 1     | 99% ± 1     |

   Above SNR 9 both methods find essentially every signal.

   ![Detection rate vs. SNR for the CNN and the drift search](figures/cnn_vs_drift_search.png)

   What this shows:
   - **Straight signals:** the drift search still clearly wins, even with
     4× more training data. So the CNN's earlier loss was not just
     undertraining.
   - **Curved signals:** the CNN wins, by a margin well outside the
     seed-to-seed spread, mostly at SNR 6-9.
   - **Wobble only:** the two are about even. Brightness changes alone
     do not hurt a search that adds up power along the whole line;
     **the bending path is what breaks the straight-line assumption.**

   Caveats: the baseline is a *straight-line* search like `turbo_seti`, so
   this shows the CNN beats the standard search, not every classical
   method; a search that also tries curved paths is the next, harder
   baseline. The straight runs used a wider drift range (±0.3 Hz/s) than
   the others (±0.2 Hz/s), so compare the curved and wobble rows with each
   other rather than with the straight rows. These numbers came from a CPU
   run; a GPU run of the same seeded notebook may differ very slightly.

## Planned next steps
- Add a classical baseline that also searches curved paths, to see whether
  the CNN still wins against a search that knows signals can bend.
- Use real noise: inject signals into actual Breakthrough Listen data
  instead of simulated chi-squared noise.
- Anomaly detection: train a model on (real) noise only and flag anything
  that does not look like noise, with no signal template at all.

## Caveats
- All results so far use simulated chi-squared noise and simple signal
  shapes (fixed width; straight, curved or wobbling paths), so they do not
  necessarily carry over to real telescope data.
- The baseline is a straight-line search. A classical search that also
  tries curved paths has not been tested yet.
- Results come from 3 seeds; more would tighten the error bars.
- `setigen` injected SNR and `turbo_seti` reported SNR are measured
  differently and are not directly interchangeable.

## Tools and data
- [Breakthrough Listen](https://seti.berkeley.edu/listen/) open data (UC Berkeley SETI Research Center)
- [blimpy](https://github.com/UCBerkeleySETI/blimpy), [turbo_seti](https://github.com/UCBerkeleySETI/turbo_seti), [setigen](https://github.com/bbrzycki/setigen)
- Python, PyTorch, NumPy, Matplotlib, Google Colab

## Running it
Open the notebook in Google Colab and run the first cell, which installs
dependencies and downloads the Voyager file (about 50 MB).

The larger experiments at the end of the notebook (item 7) train 12 models.
Switch to a GPU runtime first (Runtime → Change runtime type → T4 GPU).
They took about 53 minutes on a 4-core CPU and would take hours on Colab's
default CPU. The notebook is seeded, so a rerun should give nearly the
same numbers.

## Author
Sharvil Talware
