"""目标可视化必须忠实于所选原图和实际采样；不初始化任何硬件。"""
import json
from pathlib import Path
import sys
import tempfile
import unittest

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from unified_vln.real_decision_artifacts import backfill_run, save_decision_target
from unified_vln.local_projection import project_selected_view_target
from unified_vln.model_contract import NavigationDecisionResponse
from unified_vln.types import ViewFrame


def frame(direction="left"):
    return ViewFrame(direction, 42, 7, 1., np.full((30, 40, 3), [17, 29, 41], np.uint8),
                     np.full((30, 40), 2., np.float32),
                     np.array([[20., 0, 20.], [0, 20., 15.], [0, 0, 1.]]))


def response():
    return dict(schema_version=2, response_type="end2end_decision", session_id="offline",
                observation_id="offline_decision_000", action="NAVIGATE", direction="left",
                target="绿色瓶子", bbox_2d=[3, 4, 14, 18], waypoint=None,
                progress_analysis="test", reasoning="test")


class RealDecisionArtifactsTest(unittest.TestCase):
    def test_selected_view_and_model_box_are_preserved_without_mutating_source(self):
        f=frame();before=f.rgb.copy();raw=response()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            record=save_decision_target(root,0,raw,f)
            picture=cv2.imread(str(root/record['image_file']))
            offset=record['preview_header_height_px']
            self.assertEqual(record['source_rgb_file'],'decision_000_left.png')
            self.assertEqual(record['source_frame_id'],42)
            self.assertEqual(record['target'],'绿色瓶子')
            np.testing.assert_array_equal(picture[offset+4,3],[0,255,0])
            np.testing.assert_array_equal(picture[offset+10,30],[41,29,17])
            np.testing.assert_array_equal(f.rgb,before)
            self.assertEqual(raw,response())
            self.assertIn('绿色瓶子',(root/'targets.html').read_text())

    def test_actual_projector_pixel_and_clipping_match_saved_annotation(self):
        f=frame();raw=response();raw['bbox_2d']=[3,4,40,30]
        projection=project_selected_view_target(f,NavigationDecisionResponse.from_dict(raw)).to_dict()
        with tempfile.TemporaryDirectory() as tmp:
            record=save_decision_target(Path(tmp),0,raw,f,projection=projection,navigation_accepted=True)
            self.assertEqual(record['bbox_xyxy_original'],[3,4,40,30])
            self.assertEqual(record['bbox_xyxy_clipped'],projection['bbox_xyxy'])
            self.assertEqual(record['projection']['pixel_uv'],[21,29])
            picture=cv2.imread(str(Path(tmp)/record['image_file']))
            np.testing.assert_array_equal(picture[116+29,21],[0,255,255])

    def test_wrong_view_is_rejected_and_backtrack_does_not_invent_box_or_projection(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError,'选择方向'):
                save_decision_target(Path(tmp),0,response(),frame('forward'))
            raw={**response(),'action':'BACKTRACK','direction':None,'target':None,'bbox_2d':None}
            f=frame('forward')
            record=save_decision_target(Path(tmp),0,raw,f,navigation_accepted=False)
            self.assertIsNone(record['bbox_xyxy_clipped'])
            self.assertIsNone(record['projection'])
            picture=cv2.imread(str(Path(tmp)/record['image_file']))
            np.testing.assert_array_equal(picture[116:],cv2.cvtColor(f.rgb,cv2.COLOR_RGB2BGR))

    def test_index_keeps_all_decisions_and_escapes_model_text(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            save_decision_target(root,0,response(),frame())
            raw={**response(),'target':'瓶子<script>alert(1)</script>',
                 'stage_progress':{'current_stage':'向左走到绿色瓶子','stage_completed':1,'stage_total':2}}
            save_decision_target(root,1,raw,frame())
            page=(root/'targets.html').read_text()
            self.assertIn('decision_000_target.png',page)
            self.assertIn('decision_001_target.png',page)
            self.assertIn('向左走到绿色瓶子',page)
            self.assertNotIn('<script>',page)
            self.assertIn('&lt;script&gt;',page)

    def test_backfill_reads_saved_source_and_leaves_raw_evidence_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);raw=response();f=frame()
            path=root/'decision_000_response.json';path.write_text(json.dumps(raw,ensure_ascii=False))
            cv2.imwrite(str(root/'decision_000_left.png'),cv2.cvtColor(f.rgb,cv2.COLOR_RGB2BGR))
            (root/'decision_000_panorama_capture.json').write_text(json.dumps({'captures':{'left':{'frame_id':42}}}))
            before={p.name:p.read_bytes() for p in root.iterdir()}
            record=backfill_run(root)[0]
            self.assertEqual(record['record_mode'],'reconstructed_from_saved_observation')
            self.assertEqual(record['source_frame_id'],42)
            self.assertIsNone(record['navigation_accepted'])
            for name,data in before.items():self.assertEqual((root/name).read_bytes(),data)


if __name__=='__main__':unittest.main()
