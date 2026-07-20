import numpy as np

from instance_cir.centering import leave_one_instance_out_centers


def test_leave_one_instance_out_excludes_complete_group() -> None:
    features = np.asarray([[1.0, 0.0], [3.0, 0.0], [10.0, 2.0]], dtype=np.float32)
    centers = leave_one_instance_out_centers(features, ["a", "a", "b"])
    np.testing.assert_allclose(centers["a"], [10.0, 2.0])
    np.testing.assert_allclose(centers["b"], [2.0, 0.0])
