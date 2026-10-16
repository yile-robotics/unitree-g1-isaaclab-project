import copy
import unittest
import numpy as np
from validate_candidate import evaluate


class ValidationTests(unittest.TestCase):
    def data(self):
        candidate=dict(target_frame='torso_link',source_frame='color_optical',
                       urdf_sha256='model',probe={'tip_measured':False},board_config={'pitch':.072},
                       reference_camera={'serial':'camera'},T_torso_camera=np.eye(4).tolist())
        samples=[]
        for i,p in enumerate(([0,0,1],[.1,0,1],[0,.1,1])):
            samples.append(dict(row=i,col=i,marker_id=i,role='fit',p_camera_m=p,
                                p_torso_m=(np.array(p)+[.02,0,0]).tolist(),image=f'{i}.png',
                                metadata=dict(serial='camera',color_timestamp_ms=i,depth_timestamp_ms=i)))
        session=dict(urdf_sha256='model',probe=candidate['probe'],board_config=candidate['board_config'],
                     sampling_method='live_board_per_contact',samples=samples)
        return candidate,session

    def test_fixed_transform_exposes_bias_instead_of_refitting_it(self):
        c,s=self.data();before=copy.deepcopy(c);r=evaluate(c,s)
        self.assertAlmostEqual(r['rms_m'],.02)
        self.assertAlmostEqual(r['max_m'],.02)
        self.assertFalse(r['refitted']);self.assertEqual(c,before)
        np.testing.assert_allclose(r['mean_residual_torso_m'],[-.02,0,0])

    def test_reused_original_exposures_are_rejected(self):
        c,s=self.data()
        with self.assertRaisesRegex(ValueError,'reused'):
            evaluate(c,s,copy.deepcopy(s))

    def test_wrong_camera_and_geometry_are_rejected(self):
        c,s=self.data();s['samples'][0]['metadata']['serial']='another'
        with self.assertRaisesRegex(ValueError,'serial'):evaluate(c,s)
        c,s=self.data();s['board_config']={'pitch':.1}
        with self.assertRaisesRegex(ValueError,'board_config'):evaluate(c,s)


if __name__=='__main__':unittest.main()
