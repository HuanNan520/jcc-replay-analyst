# sample_frames

Key-frame excerpt from one complete S16 match (12 frames · covering the full flow: carousel → augment → combat → settlement).

Used by `scripts/analyze.py --frames examples/sample_frames/` as an e2e smoke test.

Frames are fed into `frame_monitor.observe` in filename order · the first frame builds the baseline · later frames trigger event extraction.

| Frame | Phase | Game round |
| --- | --- | --- |
| frame_001_pick | Carousel / opening | 1-1 |
| frame_002_pve | Minion PVE | 1-4 beach |
| frame_003_positioning | Prep / positioning | 2-x |
| frame_004_pve | Minion PVE | 2-5 |
| frame_005_augment | Augment pick | 3-2 |
| frame_006_pvp | Player combat | 3-4 |
| frame_007_pvp | Player combat | 4-1 |
| frame_008_item | Item carousel | 4-4 |
| frame_009_positioning | Prep / positioning | 5-1 |
| frame_010_pvp | Player combat | 5-4 |
| frame_011_pvp | Player combat | 6-1 |
| frame_012_end | Settlement | 2nd place |

Source frames come from the `MuMu-20260421-08xxxx` series (2560×1456 originals) · downscaled to 1280×720 and JPEG q=85 compressed to keep the size down (~2.2 MB total).
