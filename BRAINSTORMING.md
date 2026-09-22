
##### Can we remove the monotonic loss constraints/needed?

---

No, generally you should not drop a training iteration just because loss didn't decrease.


A few reasons:

- **Loss is noisy at the step level.** With mini-batch SGD, individual batches have different difficulty. A higher loss on one step doesn't mean the update was bad — it just means that batch was harder. The expected gradient over many batches is what matters.
- **Non-monotonic descent is normal and often necessary.** Optimizers with momentum (Adam, SGD+momentum) and schedulers can produce temporary loss increases that help escape saddle points or sharp minima.
- **Skipping breaks the optimizer's statistics.** Adam's running averages, learning rate schedules, and batch norm stats all assume continuous updates. Dropping iterations selectively introduces bias.

Where iteration-skipping *is* legitimate:

- **NaN/Inf gradients** — skip the update (common practice in mixed-precision training).
- **Gradient explosions beyond a threshold** — clip or skip.
- **Loss spikes far above a moving average** (some large-model training recipes do this defensively).

The right signal to watch is **validation/eval loss over a window**, not per-step training loss. If that plateaus or rises, the response is to adjust LR, regularization, data, or stop — not to discard individual steps.

---