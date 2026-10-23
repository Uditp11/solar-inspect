# Explainer 01: Segmentation

Module 1 of the pipeline. A binary panel mask for 0.1 m RGB rooftop tiles (D3, PV01), from a U-Net
written from `torch.nn` layers and trained from scratch, against a one-threshold colour rule and an
all-background null model.

Every number here traces to `docs/EXPERIMENTS.md` and a commit. The test split was read once, by
`scripts/eval_seg.py`, with `configs/seg_eval.yaml` committed in `49c44d9` before it was read, and
is not read again.

---

## What it is

A model that takes one 256 x 256 RGB tile of a rooftop orthophoto at 0.1 m per pixel and returns,
for every pixel, whether it is PV panel. The model is a U-Net of 8,633,441 parameters with no
pretrained weights, trained on the train tiles of all three PV01 rooftop subsets together (Brick,
FlatConcrete, SteelTile) and selected on val.

**What it is for: site footprint and array extent.** At 0.1 m per pixel a tile shows where the
arrays are and how far they reach, which is what this resolution is good for. **It does not compose
with the thermal stages.** PV01 is RGB rooftop imagery from one region in Jiangsu, China; modules 2
to 4 run on rendered thermal imagery (D2's drone frames of a ground-mounted plant, D1's module
crops). The mask is not registered to any
thermal frame and cannot be used to normalise thermal statistics. The Results section measures what
happens when the model is run on thermal frames anyway: it predicts nothing.

The seam:

| Component | Status | Written from scratch | Library | Cross-checked against |
|---|---|---|---|---|
| U-Net | built | ✓ (`torch.nn` layers, no pretrained weights) | none | hand-counted parameter count of the smallest net, output shapes, each skip used (`tests/test_unet.py`) |
| IoU, Dice, soft Dice loss | built | ✓ | none | hand-computed values; Dice = 2 IoU / (1 + IoU) on 50 random mask pairs (`tests/test_seg_metrics.py`) |
| Colour-threshold baseline | built | ✓ | none | the histogram search against a direct pixel count, equal (asserted in `scripts/baseline_seg.py`) |
| Tile bootstrap, boundary share | built | ✓ | `scipy.ndimage.distance_transform_edt` for the boundary distance | hand-worked cases, `tests/test_seg_evaluate.py` |

The headline, test, evaluated once: **IoU 0.9101 on Brick, 0.7910 on FlatConcrete, 0.9778 on
SteelTile**, against 0.3068, 0.4350 and 0.3912 for the colour rule on the same tiles. Every subset
is reported separately; nothing is pooled.

---

## Why this stage exists in a PV inspection pipeline

An inspection starts from a question about a site: where are the arrays, how large are they, has
anything been added or removed since the last survey. A footprint mask answers it, and it is the
denominator for per-area statements later (share of an array covered by a finding, panels per
square metre). In a pipeline that also has thermal stages, the RGB footprint is a separate product:
it would need its own registration to a thermal frame before the two could be combined, and this
project does not have co-registered data to do that.

### Semantic, not instance or panoptic

- **Semantic segmentation** labels every pixel with a class: panel or not. Two touching arrays are
  one blob.
- **Instance segmentation** labels every pixel with an object identity: this array, that array (or
  this module, that module).
- **Panoptic segmentation** does both at once: identities for countable things, classes for
  uncountable stuff (roof, road, vegetation).

Semantic is the choice here. In a regular, gridded array the instance boundary is recoverable from
geometry: once the panel area is known, rows and modules follow from the spacing, and PV01's masks
carry no per-module identity to learn from anyway. Instance segmentation is the right tool where
panels overlap in view (an oblique camera looking along a row) or the layout is not gridded, so
geometry cannot separate them.

---

## How it works

### The data and the masks

PV01 is 645 tiles of 256 x 256 px: Brick 138, FlatConcrete 413, SteelTile 94. Mask values are
exactly {0, 211, 212, 213}; the three non-zero codes are panel classes the dataset distinguishes and
this module does not. `binarise()` checks every value is one of the four and maps `mask != 0` to
panel; any other value raises rather than being counted. The share of panel pixels differs by a
factor of ten between subsets (Brick 5.20%, FlatConcrete 33.65%, SteelTile 50.41% over all tiles),
which is why every number is per subset.

All 645 tiles fit on the GPU as one uint8 tensor (127 MB), so an epoch is a permutation and some
slicing. Inputs are scaled to [0, 1] and standardised per channel with the mean and standard
deviation of the train tiles only.

### The split: by scene, linked across subsets (ADR 0005)

PV01 has no official split, and its tiles are crops of shared orthophotos. Pairs of tiles 1 or 1.41
name units apart overlap pixel for pixel (mean absolute difference 0.0 over 20,706 to 40,391 px);
pairs 2 or 3 apart touch. A random tile split would put the same pixels on both sides.

- **Scenes** are connected components of the graph "name-coordinate distance <= 4.5", which means
  touching, corners included. The graph is built over **all 645 tiles together**, not per subset:
  Brick `PV01_325573_1204537` overlaps two FlatConcrete tiles with identical pixels, and a 64-tile
  FlatConcrete scene and a 43-tile SteelTile scene are adjacent roofs (27 pairs within 4.5 units).
  Linked within each subset, those could land in different splits while a per-subset leakage check
  passed. That measurement disproved the earlier assumption, in ADR 0001, that the subsets were the
  natural scene boundary; the sentence is kept there with a retraction note.
- **SteelTile blocks.** SteelTile has only three scenes (5, 46 and 43 tiles), so whole scenes would
  give it a 5-tile val or test. Instead the 5-tile scene goes to test, a 14-tile block off one end
  of the 46-tile scene goes to val, and a 9-tile block off the end of the 43-tile roof farther from
  FlatConcrete goes to test.
- **Guard band.** Every train tile within 4.5 units of a val or test tile is relabelled `guard` and
  used for nothing: 12 tiles, all SteelTile.
- Every other scene goes to one split whole, by a seeded search for about 70 / 15 / 15 per subset.

Result, split `b5924638`: train 445, val 95, test 93, guard 12. Every val and test tile is at least
5.0 units from any train tile of any subset, where touching tiles are at most 4.24 apart. **What
leakage remains:** val and test share an orthophoto source and a region with train, and SteelTile's
val and test blocks are cut from the same roofs as its train tiles. No pixels are shared and no
tiles touch across splits; the leakage is bounded and stated, not removed.

| Subset | train | val | test | panel share train | val | test |
|---|---:|---:|---:|---:|---:|---:|
| Brick | 98 | 21 | 19 | 0.0482 | 0.0647 | 0.0578 |
| FlatConcrete | 293 | 60 | 60 | 0.2999 | 0.4123 | 0.4391 |
| SteelTile | 54 | 14 | 14 | 0.5278 | 0.5692 | 0.2547 |

The panel share shifts between splits: FlatConcrete val and test hold more panel than its train,
and SteelTile test (a 5-tile scene with little array, and a 9-tile block) holds half as much as its
train. Scores on different splits are read with that beside them.

### The U-Net

```
encoder   double conv, then 2x2 max pool; 4 times; channels 32, 64, 128, 256
bottom    double conv at 512 channels, at 16 x 16
decoder   x2 bilinear upsample, 3x3 conv halving the channels, concatenate the skip,
          double conv; 4 times, back to 256 x 256 at 32 channels
head      1x1 conv to one logit per pixel
```

A double conv is (3x3 conv, batch norm, ReLU) twice; the convs before batch norm have no bias,
because batch norm's shift makes it redundant. **8,633,441 parameters.** The smallest instance
(3 input channels, base width 4, depth 1) has 1,909, counted by hand in the test's docstring and
asserted.

**Skip connections.** Each pooling step halves the resolution, so by the bottom of the net the
features say *that* there is an array in a region, not *where its edge is* to the pixel. The skip
hands the decoder the encoder's feature map at the same scale, which still has the edge, and the
decoder draws the mask boundary from it. A test zeroes each of the skips in turn with a forward
hook and requires the output to change, so none of them is dead.

**Upsampling: bilinear then a 3x3 conv, not a transposed conv.** A stride-2 transposed conv with a
3x3 kernel covers its output pixels unevenly: alternate output pixels receive contributions from
different numbers of kernel taps, which prints a checkerboard into the output unless the weights
learn to cancel it. Bilinear interpolation spreads every input pixel evenly, and the conv after it
does the learning.

**Receptive field.** Each output pixel depends on a 252 x 252 px window of the input, measured by
back-propagating from one output pixel through a linearised copy of the net (ReLU removed, max
pooling replaced by average pooling, so no path is zeroed). That is nearly the whole 256 x 256
tile: an output pixel can in principle use context from the entire roof in view. The effective
receptive field, where the gradient is large, is smaller and was not measured.

### Loss: BCE plus soft Dice

The training loss is `BCE + soft Dice`, weights 1 and 1.

- **Binary cross-entropy** is per pixel, with a gradient of `sigmoid(logit) - target` that is
  well-behaved everywhere. On Brick, where 95% of pixels are background, it is dominated by
  background pixels, so on its own it rewards a model that predicts little panel.
- **Soft Dice** is `1 - (2 sum(p t) + eps) / (sum p + sum t + eps)` with `p = sigmoid(logit)`. It
  counts only overlap with the positive class, so the background majority does not dilute it, and
  it trains toward the overlap metric that is reported.

Dice's weakness is its gradient when prediction and target are both near-empty. With an empty
target, `sum t = 0`, the loss is `sum p / (sum p + eps)` and its gradient with respect to each `p` is
`eps / (sum p + eps)^2`. Without eps, the loss is 1 for any non-zero prediction, however small, and
0 / 0 for an exactly empty one; the gradient is either flat or undefined. Per tile, Brick tiles with
little or no panel would switch between those regimes from batch to batch. Two choices keep it
stable: Dice is **summed over the whole batch** (16 tiles), not per tile, so the target is rarely
empty; and **eps = 1** makes an empty prediction on an empty target a loss of exactly 0.

The loss is computed in float32 whatever autocast hands it, and the tests pin what eps protects
against. Without eps, a near-empty prediction (logit -20, sigmoid 2e-9 in float32) on an empty
target gives a loss of **1.0**, a correct empty mask scored as completely wrong, silently. NaN needs
an exactly-zero prediction, which float32's sigmoid only produces below a logit of about -88.5; the
test uses -200, which float16 logits can carry. With eps = 1 both cases give a finite loss of 0.

The 1 : 1 weighting was declared, not tuned, and neither term was ablated.

### IoU and Dice, and how a subset is scored

For one subset, `tp`, `fp` and `fn` are summed over **every pixel of every tile** in it, and the
metrics come from those sums:

```
IoU = tp / (tp + fp + fn)          Dice = 2 tp / (2 tp + fp + fn)
precision = tp / (tp + fp)          recall = tp / (tp + fn)
```

Not a mean of per-tile IoUs: a tile with no panel and nothing predicted has IoU 1.0 by convention
(0 / 0, nothing to find, nothing predicted), and a mean over tiles would let such tiles inflate the
score. A unit test shows the two disagree (3/4 against 1/2 on a two-tile toy). Precision with
nothing predicted and recall with nothing to find are NaN, not a number: a null model has no
precision.

**The identity.** Write `J = tp / (tp + fp + fn)`. Then

```
2 J / (1 + J) = 2 tp / (tp + fp + fn + tp) = 2 tp / (2 tp + fp + fn) = Dice
```

so Dice and IoU rank models identically and carry the same information; Dice is always the larger
of the two except at 0 and 1. A test checks the identity on 50 random mask pairs.

### Augmentation: the eight flips and rotations

Each training tile gets a random 90-degree rotation (k = 0 to 3) and then a horizontal flip with
probability 1/2, the same transform applied to the tile and its mask. Those two operations generate
the full symmetry group of a square, eight elements, each drawn with probability 1/8 (a vertical
flip is a horizontal flip plus a half turn). A nadir rooftop tile has no preferred orientation, so
every one of the eight is a tile that could have been photographed.

No photometric augmentation was used. On this RGB data colour and brightness jitter would be
legitimate: roofs and panels are photographed under different light, and the label does not depend
on absolute intensity. It was left out to change one thing at a time. **On thermal data the same
jitter would be wrong.** In a calibrated radiometric image a pixel value stands for a physical
quantity, and an anomaly is defined by its value relative to its neighbours and to the scale;
brightness jitter moves every value and contrast jitter rescales the differences, so the augmented
image no longer means what its label says. Even on 8-bit rendered thermal, where the scale is not
calibrated, the within-frame intensity contrast is the signal (it is what this project's
`delta_dn_uncalibrated` field measures).

### Training and selection

AdamW (learning rate 1e-3, weight decay 1e-4), cosine decay over 100 epochs, batch 16, float16
autocast with a gradient scaler, seed 0, cuDNN deterministic. The config (`configs/seg_unet.yaml`)
was committed before the first run. The full run took 525 s on an 8 GB laptop GPU (median 5.17 s
per epoch, peak 2,080 MiB allocated).

**The best epoch is the one with the highest mean of the three per-subset val IoUs**, declared
before the first run. Not the pooled IoU: FlatConcrete holds 60 of the 95 val tiles and would decide
it alone. The mean gives Brick, at 6% panel pixels on val, the same say as the others. Selected:
epoch 87.

How much selection flatters val, measured from the per-epoch log: the selected epoch's val IoU
against the mean of epochs 91 to 100 is **+0.0073 on Brick**, +0.0007 on FlatConcrete and +0.00001
on SteelTile. Brick's val IoU ranges from 0.8428 to 0.9074 over epochs 71 to 100, and epoch 87 is
its maximum; 21 tiles at 6% panel make it the noisy subset.

### The colour baseline and the null model

The reference frame for the U-Net's number. **Colour rule:** one threshold on R, G, B or the index
`2B - R - G`, above or below, at every integer value, chosen by the mean of the three per-subset
**train** IoUs. Chosen: **panel where 2B - R - G >= 24** (mean train IoU 0.4299; the blue channel
alone, B >= 106, reaches 0.3620). Panels are bluer than most roofs; the rule says nothing else. The
search runs on per-value histograms and cumulative sums, and the chosen rule re-applied pixel by
pixel must give the same counts, which `baseline_seg.py` asserts.

**Null model:** every pixel background. IoU 0 everywhere; pixel accuracy equals one minus the panel
share.

### Transfer learning, and why it was not used

An ImageNet-pretrained encoder (a ResNet-18 or -34 in place of the contracting path) is the
engineering call on 445 training tiles. Its early layers already respond to edges, textures and
colour contrasts, and fine-tuning needs far fewer labelled tiles than learning them does. Module 4
of this project made exactly that call for D1.

For single-channel input, such as a thermal frame, there are two ways to feed 3-channel pretrained
weights: replicate the channel three times, or sum the first conv's weights over its input-channel
axis and give it one channel. They produce the same output (`W_R x + W_G x + W_B x = (W_R + W_G +
W_B) x`) when the same normalisation is applied to each copy; summing does it at a third of the
first layer's cost. Module 4 summed the weights.

**Training this U-Net from scratch is a learning decision, not an engineering one.** The point was
to write and train the network end to end. No pretrained-encoder comparison was run, so how much
one would have gained on these tiles is not measured.

### Evaluating test once

`scripts/eval_seg.py test` is the only code that scores the test split. It refuses unless
`configs/seg_eval.yaml` and the configs it depends on (`seg_baseline.json`, `seg_unet.yaml`,
`d3_split.json`) are tracked and committed and the whole tree is clean; it refuses a checkpoint
whose SHA-256 differs from the one committed in the config, because the checkpoint is not in git;
and it refuses if any test evaluation directory already exists. It creates that directory before
reading test, so a pass that crashed still counts. The config fixed, before test was read: the
checkpoint, the threshold (sigmoid 0.5, not tuned on anything), the colour rule unchanged, the
metrics, the intervals, and which tiles the overlays would show. The same script on val reproduces
the training log's epoch-87 IoUs and the baseline's val IoUs exactly (difference 0.0).

**Intervals.** Percentile 95% bootstrap, resampling the tiles of one subset with replacement,
10,000 draws, seed 0. The per-subset metric is recomputed from the resampled tiles' summed counts,
so the point value is the reported number exactly. The U-Net and the colour rule are scored on the
**same** resampled tiles, which makes the interval of their difference paired: it reflects how the
two disagree on a tile, not how hard each draw's tiles are. **Tiles within a scene are correlated**,
so treating them as independent makes these intervals narrower than the truth. Each interval is
given with the number of spatially separate groups of tiles it rests on (connected components of
the split's tiles at the linking distance).

---

## Results

### Val (selection split), 95 tiles, epoch 87

| Subset | tiles (groups) | U-Net IoU | colour rule IoU | U-Net minus colour rule | null pixel accuracy |
|---|---|---|---|---|---:|
| Brick | 21 (8) | 0.9074 [0.8553, 0.9367] | 0.3967 [0.2538, 0.5461] | +0.5107 [0.3815, 0.6203] | 0.9353 |
| FlatConcrete | 60 (1) | 0.9272 [0.9192, 0.9337] | 0.6611 [0.6176, 0.7021] | +0.2660 [0.2269, 0.3079] | 0.5877 |
| SteelTile | 14 (1) | 0.9455 [0.9381, 0.9516] | 0.7313 [0.6757, 0.7681] | +0.2143 [0.1808, 0.2645] | 0.4308 |

FlatConcrete's 60 val tiles are a single scene and SteelTile's 14 a single block, so those two
intervals rest on one group each.

### Test, evaluated once, 93 tiles

| Subset | tiles (groups) | **U-Net IoU** | colour rule IoU | U-Net minus colour rule | null pixel accuracy |
|---|---|---|---|---|---:|
| Brick | 19 (8) | **0.9101** [0.8593, 0.9385] | 0.3068 [0.1876, 0.4177] | +0.6033 [0.5073, 0.6977] | 0.9422 |
| FlatConcrete | 60 (3) | **0.7910** [0.7350, 0.8430] | 0.4350 [0.3790, 0.4895] | +0.3560 [0.3232, 0.3944] | 0.5609 |
| SteelTile | 14 (2) | **0.9778** [0.9620, 0.9851] | 0.3912 [0.2232, 0.5849] | +0.5867 [0.3980, 0.7537] | 0.7453 |

The U-Net beats the colour rule in every subset on both splits, and no paired interval comes near
0. The null model's IoU is 0 everywhere.

U-Net, test, the other three metrics:

| Subset | Dice | precision | recall |
|---|---|---|---|
| Brick | 0.9529 [0.9243, 0.9683] | 0.9601 [0.9297, 0.9769] | 0.9459 [0.9017, 0.9706] |
| FlatConcrete | 0.8833 [0.8472, 0.9148] | 0.9703 [0.9654, 0.9750] | **0.8106** [0.7512, 0.8664] |
| SteelTile | 0.9888 [0.9806, 0.9925] | 0.9879 [0.9756, 0.9940] | 0.9897 [0.9826, 0.9929] |

On val the same model's recall is 0.9713, 0.9874 and 0.9784.

**SteelTile test is two groups** (the 5-tile scene and the 9-tile block) at 25% panel pixels, against
53% in train. Its interval describes resampling 14 tiles from two places.

### Pixel accuracy, where it misleads

On Brick, where 94 to 95% of pixels are background, pixel accuracy ranks the models wrongly. The
null model, which finds nothing, scores **0.9422** on test (0.9353 on val). The colour rule, which
finds part of the panel area (IoU 0.3068), scores **0.8950**, below the null model, because each of
its false positives costs accuracy while the panels it finds are only 6% of the pixels. The U-Net
scores 0.9946. On FlatConcrete (null 0.5609 on test) and SteelTile (0.4308 on val, 0.7453 on test)
the null model is not a strong scorer and the statement is not made there: on those subsets
accuracy happens to rank the three models in the same order as IoU.

### The panel-share shift, and why the colour rule scores higher on val than on train

The colour rule's IoU on train / val / test: Brick 0.2626 / 0.3967 / 0.3068, FlatConcrete 0.4612 /
0.6611 / 0.4350, SteelTile 0.5658 / 0.7313 / 0.3912. A fixed rule has a roughly fixed rate of false
positives per background pixel and of hits per panel pixel; `fp` then grows with the background
share and `tp` and `fn` with the panel share, so its IoU rises as the panel share rises. Val holds
more panel than train on all three subsets (Brick 0.065 against 0.048, FlatConcrete 0.41 against
0.30, SteelTile 0.57 against 0.53), and the rule scores higher on val in all three. SteelTile test,
at 25% panel, is where it falls furthest. The U-Net moves the other way on SteelTile (0.9455 val,
0.9778 test), so the shift is not one effect on both models; it means val, train and test numbers
on this split are not directly comparable, for either model.

### Where the U-Net's error is (post-hoc)

Measured after the single test evaluation, from its saved predicted masks, with no further model
pass; it changes no number above. The 3 px radius was chosen afterwards, with no measurement behind
it.

**Share of the error pixels (false positive + false negative) within 3 px of a true mask
boundary**, beside the share of all pixels that are:

| Subset (test) | FP px | FN px | errors within 3 px | FP within | FN within | all pixels within 3 px |
|---|---:|---:|---:|---:|---:|---:|
| Brick | 2,832 | 3,895 | 0.753 | 0.712 | 0.782 | 0.027 |
| FlatConcrete | 42,858 | 326,924 | 0.427 | 0.967 | 0.357 | 0.222 |
| SteelTile | 2,838 | 2,406 | 0.877 | 0.879 | 0.875 | 0.035 |

On Brick and SteelTile the remaining error is at array edges: three quarters and more of the error
pixels lie within 3 px of a boundary, where only 3% of pixels do. On FlatConcrete the false
positives are edges (97%), but **two thirds of the missed pixels lie more than 3 px inside an
array**: whole stretches of panel are not found.

**Per group of test tiles**, FlatConcrete's 60 are three groups. Two score IoU 0.9150 (8 tiles) and
0.9393 (7 tiles). The third, one 45-tile roof, scores **0.7808**, recall 0.7981, and holds 325,020 of
the subset's 326,924 missed pixels (99.4%). On that roof, large stretches of the panels render
bright, close to white, unlike the blue-grey panels elsewhere in PV01, and those stretches are what
the U-Net misses. Why the panels look like that there (glare, a different panel type, exposure) is
not established. The colour rule, which keys on blueness, also falls on FlatConcrete test (0.4350
against 0.6611 on val). The FlatConcrete test number is therefore mostly a statement about one roof.
Brick's groups range from 0.6214 (2 tiles) to 0.9475; SteelTile's are 0.9820 (the 9-tile block) and
0.9506 (the 5-tile scene).

**The overlay tiles**, chosen by the rule committed before test was read (per subset, the
lower-median and the lowest U-Net IoU): the median tiles show one rooftop array with edges off by a
pixel or two (Brick, 0.8984), rows of a large array with missed patches inside rows (FlatConcrete,
0.8845) and a large array with misses along one edge (SteelTile, 0.9707). The lowest tiles on Brick
and FlatConcrete (both IoU 0.0) are tiles whose entire labelled panel area is a sliver cut by the
tile edge, which the U-Net misses; they say little about the subsets' real error, and are shown as
declared. SteelTile's lowest (0.2919) is a small panel corner at the tile edge plus three false
positive blobs on bare ground.

### The thermal negative result

The same model, unchanged and not fine-tuned, on D2's 51 val frames (thermal, 640 x 512, rendered
to 8-bit greyscale), each replicated to three channels and standardised with PV01's train
statistics, threshold 0.5, rules committed before the run (`configs/seg_thermal_negative.yaml`). D2
has boxes, not masks, so the reference is a **filled-box mask** (pixel centres inside any
ground-truth box), an approximation that includes the gaps a box spans.

**It collapses.** Over 16,711,680 pixels it predicts **one** pixel as panel (pooled predicted
fraction 6e-8), against filled boxes covering 46% of the pixels (per frame 17% to 69%); IoU with the
filled-box mask is 0 on every frame. The one predicted pixel lies outside every box. In the three
overlay frames the panels are long bright rows on vegetated ground.

The two datasets differ in everything at once:

- **modality:** PV01 is RGB reflectance in three colour channels; D2 is a single rendered thermal
  channel, replicated, with no colour at all;
- **ground sampling distance:** PV01 is 0.1 m per pixel; D2's is not documented by the dataset
  (`docs/DATA.md` estimates about 4 cm per pixel from the sensor's pitch and focal length at an
  assumed 30 m, consistent with the measured median box of 56 x 37 px);
- **viewpoint:** both are near-nadir drone views, but PV01 tiles are 256 x 256 crops of
  orthophotos and D2 frames are whole camera frames;
- **content:** PV01 is rooftops in Jiangsu with panels dark blue-grey against brick, concrete or
  steel; D2 is a ground-mounted plant in Greece.

Which of these the collapse comes from is not separated here; a control on greyscaled PV01 tiles
would separate the loss of colour from the rest, and was not run. The result is what the scope
statement above says: this module is an RGB footprint primitive and does not carry over to the
thermal stages.

---

## Limits

- **One region, one resolution, 645 tiles.** PV01 is rooftop orthophotos from one region at 0.1 m
  per pixel. Nothing here says how the model does at another resolution, on ground-mounted plants,
  or elsewhere.
- **The intervals are optimistic.** They resample tiles as if independent; tiles of one scene are
  not. FlatConcrete val is one scene and SteelTile val one block, and SteelTile test is two groups at
  25% panel pixels against 53% in train. The intervals describe resampling those tiles, not another
  roof.
- **FlatConcrete's test number is mostly one roof.** One 45-tile roof holds 99.4% of the subset's
  missed pixels (post-hoc); the cause of its bright panel appearance is not established, and
  photometric augmentation, the obvious candidate remedy, was not tried.
- **Bounded leakage.** No pixels are shared and no tiles touch across splits, but val and test come
  from the same orthophoto source and region as train, and SteelTile's blocks from the same roofs.
- **No pretrained-encoder comparison.** The U-Net was trained from scratch by choice; the standard
  engineering alternative was not measured against it.
- **No ablation of the loss or the augmentation.** BCE + Dice at 1 : 1 and the eight-element
  flip/rotation group were declared and used, not compared with alternatives.
- **Does not transfer to thermal.** On D2 thermal frames it predicts nothing, so it cannot stand in
  for a thermal footprint.
