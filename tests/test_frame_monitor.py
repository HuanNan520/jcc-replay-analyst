"""frame_monitor 纯函数 + FrameMonitor baseline/trigger 行为。"""
import io

from PIL import Image

from src.frame_monitor import FrameMonitor, classify, dhash, hamming


class TestDHash:
    def test_identical_images_hash_equal(self, solid_frame_factory):
        b = solid_frame_factory((100, 100, 100))
        img = Image.open(io.BytesIO(b))
        assert dhash(img) == dhash(img)

    def test_two_solid_colors_both_hash_zero(self, solid_frame_factory):
        """dhash 的已知特性：纯色图相邻像素差分全为 0 · 所以 hash 也是 0。"""
        a = Image.open(io.BytesIO(solid_frame_factory((20, 20, 20))))
        b = Image.open(io.BytesIO(solid_frame_factory((220, 220, 220))))
        assert dhash(a) == 0
        assert dhash(b) == 0

    def test_stripe_vs_solid_has_high_distance(self, solid_frame_factory, stripe_frame_bytes):
        solid = Image.open(io.BytesIO(solid_frame_factory((100, 100, 100))))
        stripes = Image.open(io.BytesIO(stripe_frame_bytes))
        d = hamming(dhash(solid), dhash(stripes))
        assert d > 10

    def test_hamming_zero_for_same(self):
        assert hamming(0xFFFF, 0xFFFF) == 0
        assert hamming(0, 0) == 0

    def test_hamming_counts_bit_diff(self):
        assert hamming(0b1010, 0b0101) == 4
        assert hamming(0b0000, 0b1111) == 4
        assert hamming(0b1100, 0b1010) == 2


class TestFrameMonitorObserve:
    def test_first_frame_all_baseline_no_trigger(self, tiny_png_bytes):
        mon = FrameMonitor(screen_size=(2560, 1456))
        events = mon.observe(tiny_png_bytes)
        assert len(events) > 0
        assert all(not e.triggered for e in events)

    def test_second_identical_frame_no_trigger(self, tiny_png_bytes):
        mon = FrameMonitor(screen_size=(2560, 1456))
        mon.observe(tiny_png_bytes)
        events = mon.observe(tiny_png_bytes)
        assert not any(e.triggered for e in events)
        assert all(e.distance == 0 for e in events)

    def test_reset_clears_baseline(self, tiny_png_bytes):
        mon = FrameMonitor(screen_size=(2560, 1456))
        mon.observe(tiny_png_bytes)
        mon.reset()
        assert mon.frame_count == 0
        events = mon.observe(tiny_png_bytes)
        assert all(not e.triggered for e in events)

    def test_orientation_auto_picks_landscape(self):
        mon = FrameMonitor(screen_size=(2560, 1456))
        assert mon.orientation == "landscape"

    def test_orientation_auto_picks_portrait(self):
        mon = FrameMonitor(screen_size=(1080, 2400))
        assert mon.orientation == "portrait"

    def test_frame_count_increments(self, tiny_png_bytes):
        mon = FrameMonitor(screen_size=(2560, 1456))
        mon.observe(tiny_png_bytes)
        mon.observe(tiny_png_bytes)
        mon.observe(tiny_png_bytes)
        assert mon.frame_count == 3

    def test_changed_regions_filter(self, tiny_png_bytes):
        mon = FrameMonitor(screen_size=(2560, 1456))
        events = mon.observe(tiny_png_bytes)
        assert mon.changed_regions(events) == []


class TestClassify:
    def test_empty_changes_is_idle(self):
        assert classify([]) == "idle"

    def test_popup_needs_center_plus_two_major(self):
        assert classify(["center_popup", "hud_top", "shop_bottom"]) == "popup"

    def test_center_only_is_board_motion_not_popup(self):
        assert classify(["center_popup"]) == "board_motion"

    def test_shop_plus_hud_is_trade(self):
        assert classify(["shop_bottom", "hud_top"]) == "trade"

    def test_shop_only_is_refresh(self):
        assert classify(["shop_bottom"]) == "shop_refresh"

    def test_hud_only_is_hud_change(self):
        assert classify(["hud_top"]) == "hud_change"

    def test_carry_zone_only_is_board_motion(self):
        assert classify(["carry_zone"]) == "board_motion"

    def test_bench_only_is_bench_change(self):
        assert classify(["bench_row"]) == "bench_change"
