# SETI Narrowband Signal Detection: Classical Search vs. Machine Learning

An undergraduate research project exploring how well a standard SETI search
algorithm detects faint drifting radio signals, and whether a small neural
network can do better.

**Status:** in progress. Early experiments only, so results are preliminary.

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

   Note: the CNN numbers vary from run to run because the noise and the
   training order are not seeded yet. The cutoff table above and this
   comparison come from different runs.

## Planned next steps
- Fix random seeds and repeat trials to get averaged results with error bars.
- Test both methods on signals where a straight-line search should struggle:
  curved (non-linear) drift, broader or intermittent signals, and
  interference.
- Test on real archive data.

## Caveats
- The synthetic data is simple (one signal type, fixed width), so results do
  not necessarily carry over to real telescope data.
- `setigen` injected SNR and `turbo_seti` reported SNR are measured
  differently and are not directly interchangeable.

## Tools and data
- [Breakthrough Listen](https://seti.berkeley.edu/listen/) open data (UC Berkeley SETI Research Center)
- [blimpy](https://github.com/UCBerkeleySETI/blimpy), [turbo_seti](https://github.com/UCBerkeleySETI/turbo_seti), [setigen](https://github.com/bbrzycki/setigen)
- Python, PyTorch, NumPy, Matplotlib, Google Colab

## Running it
Open the notebook in Google Colab and run the first cell, which installs
dependencies and downloads the Voyager file (about 50 MB).

## Author
Sharvil Talware
