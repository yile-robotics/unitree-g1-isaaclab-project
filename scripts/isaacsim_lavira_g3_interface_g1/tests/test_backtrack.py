from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from unified_vln.backtrack import (  # noqa: E402
    build_stored_reverse_route,
    next_route_checkpoint_index,
)
from unified_vln.odometry import Pose2D  # noqa: E402
from unified_vln.model_client import (  # noqa: E402
    CompletedWaypoint,
    build_model_history,
    select_model_history_records,
)


def _record(waypoint_id: int, start_x: float, end_x: float):
    return SimpleNamespace(
        waypoint_id=waypoint_id,
        decision_pose=Pose2D(start_x, 0.0, 0.0, float(waypoint_id)),
        arrival_pose=Pose2D(end_x, 0.0, 0.0, float(waypoint_id) + 0.5),
        executed_world_path_xy=np.array(
            [[start_x, 0.0], [(start_x + end_x) / 2.0, 0.0], [end_x, 0.0]],
            dtype=np.float64,
        ),
    )


class StoredReverseRouteTest(unittest.TestCase):
    def test_truncated_wire_history_keeps_explicit_local_record_mapping(self):
        image = np.zeros((2, 2, 3), dtype=np.uint8)
        records = [
            CompletedWaypoint(
                waypoint_id=index,
                decision_step=index,
                direction="forward",
                target=f"target {index}",
                init_rgb=image,
                direction_rgb=image,
            )
            for index in range(6)
        ]

        selected = select_model_history_records(records, max_waypoints=4)
        history, _ = build_model_history(records, max_waypoints=4)

        self.assertEqual([record.waypoint_id for record in selected], [2, 3, 4, 5])
        self.assertEqual([entry.waypoint_id for entry in history], [0, 1, 2, 3])
        self.assertEqual(selected[1].waypoint_id, 3)

    def test_reverses_all_successful_segments_through_target_observation(self):
        route = build_stored_reverse_route(
            [_record(0, 0.0, 1.0), _record(1, 1.0, 2.0)],
            target_waypoint_id=0,
            current_pose=Pose2D(2.0, 0.0, 0.0, 3.0),
            max_start_drift_m=0.25,
            max_path_length_m=6.0,
        )

        np.testing.assert_allclose(route.points_world_xy[0], [2.0, 0.0])
        np.testing.assert_allclose(route.points_world_xy[-1], [0.0, 0.0])
        self.assertAlmostEqual(route.path_length_m, 2.0)

    def test_latest_waypoint_returns_to_its_decision_pose(self):
        route = build_stored_reverse_route(
            [_record(0, 0.0, 1.0), _record(1, 1.0, 2.0)],
            target_waypoint_id=1,
            current_pose=Pose2D(2.0, 0.0, 0.0, 3.0),
            max_start_drift_m=0.25,
            max_path_length_m=6.0,
        )

        np.testing.assert_allclose(route.target_world_xy, [1.0, 0.0])
        self.assertAlmostEqual(route.path_length_m, 1.0)

    def test_rejects_stale_current_pose(self):
        with self.assertRaisesRegex(ValueError, "stale"):
            build_stored_reverse_route(
                [_record(0, 0.0, 1.0)],
                target_waypoint_id=0,
                current_pose=Pose2D(3.0, 0.0, 0.0, 2.0),
                max_start_drift_m=0.25,
                max_path_length_m=6.0,
            )

    def test_checkpoint_uses_bounded_polyline_lookahead(self):
        points = np.array(
            [[0.0, 0.0], [0.4, 0.0], [0.8, 0.0], [1.2, 0.0]],
            dtype=np.float64,
        )
        self.assertEqual(
            next_route_checkpoint_index(
                points, current_index=0, segment_length_m=0.75
            ),
            2,
        )
        self.assertEqual(
            next_route_checkpoint_index(
                points, current_index=2, segment_length_m=0.75
            ),
            3,
        )


if __name__ == "__main__":
    unittest.main()
