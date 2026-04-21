"""_coerce_world_state 边界测试 · 验证脏 VLM 输出被强行规范成合法 WorldState。"""
from src.schema import WorldState
from src.vlm_client import _coerce_world_state


class TestCoerceWorldState:
    def test_valid_full_payload(self):
        payload = {
            "stage": "pvp", "round": "3-2", "hp": 72, "gold": 28,
            "level": 7, "exp": "12/20",
            "board": [{"name": "芸阿娜", "star": 2, "items": ["无尽之刃"]}],
            "bench": [], "bag": [{"slot": 0, "name": "暴风大剑"}],
            "shop": ["诺克撒尔", "加里奥", "卡莎", "梅尔九五", "弗雷神谕"],
            "active_traits": [{"name": "法师", "count": 4, "tier": "silver"}],
            "augments": [], "opponents_preview": [],
        }
        ws = _coerce_world_state(payload)
        assert isinstance(ws, WorldState)
        assert ws.stage == "pvp"
        assert ws.round == "3-2"
        assert ws.hp == 72
        assert ws.gold == 28
        assert ws.level == 7
        assert len(ws.board) == 1
        assert ws.board[0].name == "芸阿娜"
        assert ws.board[0].star == 2
        assert ws.board[0].items == ["无尽之刃"]
        assert len(ws.shop) == 5
        assert ws.active_traits[0].tier == "silver"

    def test_missing_required_fields_get_defaults(self):
        ws = _coerce_world_state({})
        assert ws.stage == "unknown"
        assert ws.round == "?-?"
        assert ws.hp == 0
        assert ws.gold == 0
        assert ws.level == 1
        assert ws.exp == "0/0"
        assert ws.board == []
        assert ws.bench == []
        assert ws.bag == []
        assert ws.shop == []
        assert ws.active_traits == []
        assert ws.augments == []
        assert ws.opponents_preview == []

    def test_garbage_stage_falls_back_to_unknown(self):
        ws = _coerce_world_state({"stage": "completely_invalid"})
        assert ws.stage == "unknown"

    def test_hp_clamped_to_range(self):
        ws_high = _coerce_world_state({"hp": 999})
        assert ws_high.hp == 100
        ws_neg = _coerce_world_state({"hp": -50})
        assert ws_neg.hp == 0

    def test_level_clamped_to_range(self):
        assert _coerce_world_state({"level": 99}).level == 10
        assert _coerce_world_state({"level": 0}).level == 1

    def test_non_numeric_hp_falls_back_to_default(self):
        ws = _coerce_world_state({"hp": "abc"})
        assert ws.hp == 0

    def test_unit_without_name_dropped(self):
        payload = {"board": [{"name": "芸阿娜", "star": 2}, {"star": 3}]}
        ws = _coerce_world_state(payload)
        assert len(ws.board) == 1
        assert ws.board[0].name == "芸阿娜"

    def test_unit_star_clamped(self):
        payload = {"board": [{"name": "x", "star": 99}]}
        ws = _coerce_world_state(payload)
        assert ws.board[0].star == 3

    def test_trait_invalid_tier_falls_back(self):
        payload = {"active_traits": [{"name": "法师", "count": 4, "tier": "wtf"}]}
        ws = _coerce_world_state(payload)
        assert ws.active_traits[0].tier == "none"

    def test_position_malformed_becomes_none(self):
        payload = {"board": [{"name": "x", "star": 1, "position": "garbage"}]}
        ws = _coerce_world_state(payload)
        assert ws.board[0].position is None

    def test_position_valid_pair_preserved(self):
        payload = {"board": [{"name": "x", "star": 1, "position": [2, 3]}]}
        ws = _coerce_world_state(payload)
        assert ws.board[0].position == (2, 3)

    def test_shop_drops_empty_entries(self):
        payload = {"shop": ["a", "", None, "b"]}
        ws = _coerce_world_state(payload)
        assert ws.shop == ["a", "b"]

    def test_opponent_hp_clamped(self):
        payload = {"opponents_preview": [{"hp": 500, "top_carry": "金克丝"}]}
        ws = _coerce_world_state(payload)
        assert ws.opponents_preview[0].hp == 100
        assert ws.opponents_preview[0].top_carry == "金克丝"
