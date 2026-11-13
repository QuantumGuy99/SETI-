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
5. **ML model:** training a small convolutional network to classify them
   (in progress).

## Planned next steps
- Compare the model against `turbo_seti` on the same data, including false
  alarm rates.
- Repeat trials to get detection-rate curves instead of single runs.
- Test on more realistic signals and on real archive data.

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
