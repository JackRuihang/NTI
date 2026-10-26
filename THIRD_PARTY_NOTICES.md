# License scope and third-party notices

The original NTI contributions and the NTI-S/B/L 250K-step generator inference
checkpoints released by this project are provided under the [MIT License](LICENSE).
This does not relicense third-party code, tokenizers, datasets, or evaluation assets,
and does not grant rights to assets that are downloaded separately.

## reAR

This project uses reAR's autoregressive modeling conventions and imports its frozen
MaskGIT tokenizer implementation from a separate checkout. The upstream project is
[QY-H00/reAR](https://github.com/QY-H00/reAR), pinned in our instructions to commit
`61cc871cc188e4593160156480a10b052eececdb`. reAR is licensed under Apache-2.0;
any reAR-derived portions retain that license, reproduced in
[licenses/reAR-Apache-2.0.txt](licenses/reAR-Apache-2.0.txt).
NTI-specific training, replay supervision, configurations, and release documentation
are additions or modifications for this project, not the unmodified upstream code.

The tokenizer weights are not included in our generator checkpoint release.
Obtain them separately and follow their upstream terms.

## Other external assets

ImageNet data, the ADM evaluator, its reference batches, and installed Python
dependencies retain their respective licenses and terms. They are not covered by
the NTI MIT grant and are not redistributed in the checkpoint release.
