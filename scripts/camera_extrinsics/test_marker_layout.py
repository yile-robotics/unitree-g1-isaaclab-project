"""Verify layout checks without camera or robot access."""
import unittest
from unittest.mock import patch
import cv2
import numpy as np
from verify_marker_layout import detect, check_grid, Evidence, marker_center

GRID = [[6*c+r for c in range(6)] for r in range(6)]


class MarkerLayoutTests(unittest.TestCase):
    def markers(self):
        result = {}
        for i in (0, 1, 2, 6, 7, 8, 12, 13, 14):
            x, y = 100+40*(i//6), 80+40*(i%6)
            result[i] = np.array([[x-12,y-12],[x+12,y-12],[x+12,y+12],[x-12,y+12]],np.float32)
        return result

    def test_decodes_marker_without_expected_id(self):
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
        image = np.full((160,160),255,np.uint8)
        image[30:130,30:130] = cv2.aruco.generateImageMarker(dictionary, 19, 100,borderBits=2)
        self.assertEqual(set(detect(image)), {19})
        image[30:130,30:130] = cv2.aruco.generateImageMarker(dictionary, 19, 100,borderBits=1)
        self.assertEqual(set(detect(image,border_bits=1)), {19})

    def test_accepts_grid_and_rejects_swapped_ids(self):
        markers = self.markers()
        accepted, H, error = check_grid(markers, GRID)
        self.assertEqual(set(accepted), set(markers))
        self.assertIsNone(error)
        markers[0], markers[14] = markers[14], markers[0]
        accepted, _, error = check_grid(markers, GRID)
        self.assertFalse(accepted)
        self.assertIsNotNone(error)

    def test_strong_perspective_grid_uses_projective_centres(self):
        H=np.array([[200.,10,400],[0,100,100],[.7,.25,1]])
        offsets=np.array([[-.38,-.38],[.38,-.38],[.38,.38],[-.38,.38]])
        markers={}
        for i in (0,1,2,6,7,8,12,13,14):
            obj=offsets+np.array([i//6,i%6])
            markers[i]=cv2.perspectiveTransform(obj.reshape(1,4,2),H).reshape(4,2)
        self.assertGreater(max(np.linalg.norm(c.mean(0)-marker_center(c)) for c in markers.values()),2)
        accepted,_,error=check_grid(markers,GRID)
        self.assertIsNone(error)
        self.assertEqual(set(accepted),set(markers))

    def test_upscaled_only_detection_preserves_original_pixel_coordinates(self):
        original=np.array([[10.,10.],[30,10],[30,30],[10,30]],np.float32)
        for scale,shift in ((2,.5),(3,1.)):
            class Detector:
                def detectMarkers(self,image):
                    if image.shape[0]!=60*scale:
                        return [],None,[]
                    return [(original*scale+shift).reshape(1,4,2)],np.array([[19]]),[]
            with patch.object(cv2.aruco,'ArucoDetector',return_value=Detector()):
                observed=detect(np.full((60,60),255,np.uint8))
            np.testing.assert_allclose(observed[19],original,atol=1e-6)

    def test_rejects_unknown_or_insufficient_markers(self):
        markers = self.markers()
        markers[100] = markers[0]
        self.assertFalse(check_grid(markers,GRID)[0])
        self.assertFalse(check_grid({i:markers[i] for i in (0,1,2)},GRID)[0])

    def test_requires_distinct_frames_and_counts_only_observed_ids(self):
        evidence = Evidence(GRID)
        evidence.add(5, {0:None,6:None})
        evidence.add(5, {0:None,6:None,12:None})
        self.assertEqual(evidence.counts[0],1)
        self.assertEqual(evidence.counts[12],0)
        evidence.add(6,{0:None})
        self.assertEqual(evidence.counts[0],2)
        self.assertEqual(evidence.counts[6],1)


if __name__ == '__main__':
    unittest.main()
